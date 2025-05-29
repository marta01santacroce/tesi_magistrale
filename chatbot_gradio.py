import gradio as gr  #versione 5.30.0
import DB
import search_v2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage
from openai import OpenAI
import os
from pathlib import Path
import re
from dotenv import load_dotenv
from urllib.parse import quote
from collections import defaultdict
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register
import save_embeddings_recursive as sv_rec
import update_name_pdf_and_move_folder as rename_move
import subprocess
import warnings
import docker

warnings.filterwarnings("ignore")
 

# docker connessione
client_docker = docker.from_env()

container = client_docker.containers.get("postgres_pgvector")

if container.status != "running":
    print(f"Avvio del container {container.name}...")
    container.start()
else:
    print(f"Il container {container.name} è già in esecuzione.")

 
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
# Load environment variables from .env file
load_dotenv(override=True)
# Retrieve your API key from your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

CHUNK_SIZE=512

TABLE_NAME = 'embeddings_recursive' # nome della tabella da cui effettuare la ricerca ibrida -> 
# CAMBIARE IL NOME DELLA TABELLA DI INTERESSE!!!!! -> 
    # embeddings -> usa recursive character splitter
    # embeddings_recursive -> usa recursive character splitter v2
    # embeddings_character_splitter -> usa character splitter
    # embeddings_semantic_splitter_percentile -> usa semantic splitter con percentile come breakpoint_threshold_type

no_docs_template = (
    "I'm sorry but I could not find any relevant documents to answer your query.\n"
    "Try rephrasing the query or ask me something more specific.\n" 
    "If you want a more in-depth explanation of the previous answer, take up the question you asked and ask me explicitly."
)


# Inizializza la memoria della conversazione
message_history = ChatMessageHistory()

# RAG Prompt Template with context
prompt_rag_context = [
    {"role": "developer", 
     "content": "You are an AI assistant specialized in answering questions based on the provided context.\n"
    "Follow these guidelines:\n"
    "- Use only the given context to generate your answer.\n"
    "- Provide a clear and relevant response. Avoid unnecessary details.\n"
    "- Do NOT mention the context, sources, or any document references in your response.\n"
    "- If you cannot answer based on the context, reply with exactly the following message:\n"
            "\"I'm sorry but I could not find any relevant documents to answer your query.\n"
            "Try rephrasing the query or ask me something more specific.\n"
            "If you want a more in-depth explanation of the previous answer, take up the question you asked and ask me explicitly.\"\n"
            
    },
    {"role": "user", 
     "content": "Previous conversation:\n{chat_history}\n---\nHere is the relevant context:\n{context}\n---\nNow, answer the following question based strictly on the context and on previous conversation .\n\nQuestion: {question}"},
]

'''
# RAG Prompt Template no context
prompt_rag_no_context = [
    {"role": "developer", "content": "You are an AI assistant specialized in answering questions based on the past history.Provide a clear and relevant response. Avoid unnecessary details.\n"},
    {"role": "user", "content": "Previous conversation:\n{chat_history}\n---\nNow, answer the following question based on previous conversation and your konwladge .\n\nQuestion: {question}"}
]
'''
def normalize(text):
    return " ".join(text.strip().lower().split())

def clean_snippet(text):
    text = text.replace("\n", " ")
    text = re.sub(r'([a-zA-Z]+)(\d+)', r'\1 \2', text)
    text = text.lower()
    text = text.replace('- ','')
    text = text[3:-3]
    return text.strip()

def double_url_encode(s):
    return quote(quote(s, safe=''), safe='')

def gpt_generate(chat_history, question, prompt, context=""):

    formatted_prompt = [
        {"role": msg["role"], "content": msg["content"].format(context=context, chat_history=chat_history, question=question)}
        for msg in prompt
    ]
    completion = client.chat.completions.create(
        model = "gpt-4o-mini",
        temperature = 0.2,
        messages = formatted_prompt
    )

    generated_text = completion.choices[0].message.content

    return generated_text



def retrieve_documents(query):

    # Load database connection
    cursor, conn = DB.connect_db(
        host = "localhost",
        database = "rag_db",
        user = "rag_user",
        password = "password",
        port = "5432"
    )
    """Recupera i documenti più rilevanti dal database basandosi sulla query dell'utente."""

    # Genera l'embedding della query
    formatted_query = f"query: {query}"
    query_embedding = embedding_model.embed_query(formatted_query)
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    # Esegue la ricerca nel database (BM25 + Similarità Coseno + reranking)
    results = search_v2.hybrid_search(cursor, query, query_embedding_str,table_name=TABLE_NAME)

    # Converte i risultati in oggetti Document di LangChain
    documents = [
        Document(
            page_content = res[1],
            metadata = {
                "source": res[0],
                "page": res[2],
                "bm25_score": res[3],
                "semantic_score": res[4],
                "hybrid_score": res[5],  # 0.3 * BM25 + 1 * Semantico
            }
        )
        for res in results
    ]

    DB.close_db(cursor, conn)

    return documents


def explain_relevance(source_to_contents, query):
    explanation_prompt = [{"role": "system", "content": "You are a helpful assistant that explains why each document was considered relevant.For each document you answer with: 'Document: file_name.pdf explanation'"}]
    
    user_content = "Given the user query:\n\n" + query + "\n\nExplain why each document is relevant:\n"

    for source, contents in source_to_contents.items():
        combined = "\n".join(contents[:3])  # Max 3 chunk per doc per brevità
        user_content += f"\nDocument: {source}\nContents:\n{combined}\n---"
    
    explanation_prompt.append({"role": "user", "content": user_content})
    
    completion = client.chat.completions.create(
        model = "gpt-4o-mini",
        temperature = 0.2,
        messages = explanation_prompt
    )
    
    return completion.choices[0].message.content


# Chatbot function
def chatbot_response(query, history):
    """Gestisce la risposta del chatbot. Ignora input vuoti."""
    if not query.strip():
        return [{"role": "assistant", "content": "⚠️ Empty user message. Please write a question for me!"}]
    
    docs = retrieve_documents(query)
    retrieved_docs_text = [doc.page_content for doc in docs]
    context = "\nExtracted documents:\n" + "".join([f"Document {str(i)}\n" + doc for i, doc in enumerate(retrieved_docs_text)])

    # Get conversation history
    chat_history = "\n".join([
        f"user: {msg.content}" if isinstance(msg, HumanMessage) else f"assistant: {msg.content}"
        for msg in message_history.messages
    ]) or "No previous conversation."

    total_answer = ""




    
    if retrieved_docs_text:
        answer = gpt_generate(chat_history, query, prompt_rag_context, context)

        if normalize(answer) != normalize(no_docs_template):

            # Raggruppa i contenuti per source
            source_to_contents = defaultdict(list)
            for doc in docs:
                source_to_contents[doc.metadata["source"]].append(doc.page_content)

            # Riorganizza le fonti per documento con link per ogni pagina
            doc_pages = defaultdict(list)
            for doc in docs:
                doc_name = doc.metadata["source"]
                page = doc.metadata["page"]
                snippet = clean_snippet(doc.page_content[:CHUNK_SIZE])
                doc_pages[doc_name].append((page, snippet))


            # Ottieni la spiegazione dei documenti
            explanation_text = explain_relevance(source_to_contents, query)
            explanation_map = {}  # Mappa {doc_name -> spiegazione}
            current_doc = None

            # Estrai le spiegazioni per ciascun documento
            for line in explanation_text.split("\n"):
                # Cerca una riga che contiene "Document:" o "Documento:", anche con simboli o Markdown
                match = re.search(r"[Dd]ocument(?:o)?:\s*([^\*\n]+\.pdf)", line)
                if match:
                    current_doc = match.group(1).strip()
                    explanation_map[current_doc] = ""
                elif current_doc:
                    explanation_map[current_doc] += line.strip() + " "
            
            

            sources_text = ""
            for doc_name, pages in doc_pages.items():
                sorted_pages = sorted(set(pages), key=lambda x: x[0])
                page_links = ", ".join([
                    f"[pag.{page}](http://localhost:8080/viewer.html?file={quote(doc_name)}&page={page}&highlight={double_url_encode(snippet)})"
                    for page, snippet in sorted_pages
                ])
                title_document = doc_name[:70] + "..."
                explanation = explanation_map.get(doc_name, "Nessuna spiegazione disponibile.")
                explanation=re.sub(r'^[^a-zA-Z]*', '', explanation)
                sources_text += f"🔹 {title_document}: {page_links}\n🔸 {explanation}\n\n"


            total_answer = answer + "\n\n" + sources_text
            
        else:
          total_answer = answer  
    else:
        #answer = gpt_generate(chat_history, query, prompt_rag_no_context, context)
        # Template di risposta nel caso non venga trovato alcun documento
        #total_answer = answer

        total_answer = no_docs_template
        answer = total_answer

    # Update message history
    message_history.add_user_message(query)
    message_history.add_ai_message(answer)

    return [{"role": "assistant", "content": total_answer}]


def upload_and_process_files(file_list):
    # Load database connection
    cursor, conn = DB.connect_db(
        host = "localhost",
        database = "rag_db",
        user = "rag_user",
        password = "password",
        port = "5432"
    )
    
    if not file_list:
        return "⚠️ Nessun file selezionato."

    
    output_elaborati = []
    output_non_elaborati = []

    for file_obj in file_list:
        
        value = 0
        
        file_path = Path(file_obj.name)
        file_name = os.path.basename(file_obj.name)
             
        try:
                                
            value = sv_rec.save_pdfs(file_path=file_path, option='N', cursor=cursor, table_name=TABLE_NAME,embedding_model=embedding_model, conn=conn)

            if value is None:
                output_non_elaborati.append(str(file_name))
                
            else:
                output_elaborati.append(str(file_name))
                base_path = os.getcwd()  # path corrente
                subfolder = "pdf_files" # nome cartella dei pdfs
                destination_folder = os.path.join(base_path, subfolder)
                rename_move.rename_and_move_single_pdf(file_path = file_path, destination_folder = destination_folder)
                    
        except Exception as e:
            output_non_elaborati.append(str(file_name))
            

    DB.save_changes(conn)
    DB.close_db(cursor, conn)
   
    return  (
            f"✅ {len(output_elaborati)} file PDF elaborati:\n"
            + "\n".join(output_elaborati)
            + f"\n\n⛔ {len(output_non_elaborati)} file PDF NON elaborati perché già presenti nel DB:\n"
            + "\n".join(output_non_elaborati)
            )



with gr.Blocks(fill_height=True) as chatbot_ui:
    
    gr.Markdown("<h2 style='text-align: center;'> GenderMoRe Chatbot</h2>")
    gr.ChatInterface(

        fn = chatbot_response,
        fill_height=True,fill_width=True,
        type = 'messages',
        show_progress ='minimal',
        save_history = True,
        flagging_mode = "manual"
        #atoscroll=True

    )

with gr.Blocks(fill_height=True) as upload_ui:
    
    gr.Markdown("<h2 style='text-align: center;'>Caricamento Documenti</h2>")

    file_input = gr.File(
        file_types = [".pdf"],
        file_count = "multiple",
        label="Inserisci i PDF",
        height="height:50%"
    )

    output_box = gr.Textbox(
        label="🔔 Stato caricamento",
        lines=15,
        max_lines=15,
        show_copy_button=True
    )

    with gr.Row():
        clear_button = gr.Button(value="Clear all")
        upload_button = gr.Button(value="Submit")
        

    upload_button.click(
        fn=upload_and_process_files,
        inputs=file_input,
        outputs=output_box
    )

    def clear_all():
        return None, ""

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[file_input, output_box]
    )

if __name__ == "__main__":

    # Imposta la cartella da servire
    pdf_dir = os.path.join(os.getcwd(), "pdf_files")

    # Avvia un server HTTP in background sulla cartella pdf_files sulla porta 8080
    subprocess.Popen(
        ["python", "-m", "http.server", "8080"],
        cwd=pdf_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    
    #chatbot_ui.launch()
    with gr.Blocks(
        
        theme='allenai/gradio-theme', 
        fill_height=True,
        fill_width=True, 
        title="GenderMoRe Chatbot",
        css="""
                    html, body, #root, .gradio-container {
                        height: 100% !important;
                        margin: 0;
                        padding: 0;
                    }
                    .gradio-container {
                        display:flex;
                        flex-direction: column;
                    }
                    main {
                        flex-grow: 1;
                        overflow: auto;
                    }
                    """) as demo:
        gr.TabbedInterface(
            [chatbot_ui, upload_ui],
            ["🤖 Chatbot", "📂 Carica Documenti"]
        )
        demo.load(concurrency_limit=None)

    demo.launch()

        
