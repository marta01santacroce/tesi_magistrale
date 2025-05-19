
import gradio as gr  #versione 5.20.1
import DB
import search_v2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage
from openai import OpenAI
import os

import re
from dotenv import load_dotenv
from urllib.parse import quote
from collections import defaultdict
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register
import warnings
warnings.filterwarnings("ignore")
 
tracer_provider = register(
    project_name="prova_marta",
    endpoint="http://localhost:6006/v1/traces",
)
 
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Load environment variables from .env file
load_dotenv(override=True)

# Retrieve your API key from your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print('Chiave usata: '+ OPENAI_API_KEY)
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
    "Try rephrasing the query or ask me something more specific."
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
            "Try rephrasing the query or ask me something more specific.\"\n"
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
                #"title": res[6]
            }
        )
        for res in results
    ]

    return documents


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

            # Riorganizza le fonti per documento con link per ogni pagina
            doc_pages = defaultdict(list)
            for doc in docs:
                doc_name = doc.metadata["source"]
                page = doc.metadata["page"]
                snippet = clean_snippet(doc.page_content[:CHUNK_SIZE])
                doc_pages[doc_name].append((page, snippet))


            sources_text = ""
            for doc_name, pages in doc_pages.items():
                sorted_pages = sorted(set(pages), key=lambda x: x[0])
                page_links = ", ".join([
                    f"[pag.{page}](http://localhost:8080/viewer.html?file={quote(doc_name)}&page={page}&highlight={double_url_encode(snippet)})"
                    for page, snippet in sorted_pages
                ])
                title_document=doc_name[:70]+"..."
                sources_text += f"🔹 {title_document}: {page_links}\n"

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

        

# Gradio Interface
chatbot_ui = gr.ChatInterface(

    fn = chatbot_response,
    type = 'messages',
    title = "🕷️RAG Chatbot🕷️",
    theme ='allenai/gradio-theme',
    show_progress ='full',
    fill_height = True,
    save_history = True,
    flagging_mode = "manual",
    css="""
        * {
            font-family: 'Roboto', monospace;
        }
        .prose {
            font-family: 'Roboto', monospace;
        }
    """

)

if __name__ == "__main__":
        
    # Load database connection
    cursor, conn = DB.connect_db(
        host = "localhost",
        database = "rag_db",
        user = "rag_user",
        password = "password",
        port = "5432")
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    
    chatbot_ui.launch()
