
import gradio as gr  #versione 5.20.1
import DB
import search_v2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import torch
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage
from openai import OpenAI
import os
from dotenv import load_dotenv
from urllib.parse import quote

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

TABLE_NAME = 'embeddings' # nome della tabella da cui effettuare la ricerca ibrida -> 
# CAMBIARE IL NOME DELLA TABELLA DI INTERESSE!!!!! -> 
    # embeddings -> usa recursive character splitter
    # embeddings_character_splitter -> usa character splitter
    # embeddings_semantic_splitter_percentile -> usa semantic splitter con percentile come breakpoint_threshold_type



# Inizializza la memoria della conversazione
message_history = ChatMessageHistory()

# RAG Prompt Template with context
prompt_rag_context = [
    {"role": "developer", "content": "You are an AI assistant specialized in answering questions based on the provided context.\nFollow these guidelines:\n- Use only the given context to generate your answer.\n- Provide a clear and relevant response. Avoid unnecessary details.\n- Do NOT mention the context, sources, or any document references in your response."},
    {"role": "user", "content": "Previous conversation:\n{chat_history}\n---\nHere is the relevant context:\n{context}\n---\nNow, answer the following question based strictly on the context and on previous conversation .\n\nQuestion: {question}"},
]
# RAG Prompt Template no context
prompt_rag_no_context = [
    {"role": "developer", "content": "You are an AI assistant specialized in answering questions based on the past history.Provide a clear and relevant response. Avoid unnecessary details.\n"},
    {"role": "user", "content": "Previous conversation:\n{chat_history}\n---\nNow, answer the following question based on previous conversation and your konwladge .\n\nQuestion: {question}"}
]

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
                "hybrid_score": res[5]  # 0.3 * BM25 + 1 * Semantico
            }
        )
        for res in results
    ]

    return documents


# Chatbot function
def chatbot_response(query,history):
    """Gestisce la risposta del chatbot. Ignora input vuoti."""
    if not query.strip():  # Evita invii di stringhe vuote
        return [{"role": "assistant", "content": "⚠️ Empty user message. Please write a question for me!"}]
    
    docs = retrieve_documents(query)
    retrieved_docs_text = [doc.page_content for doc in docs]
    context = "\nExtracted documents:\n" + "".join([f"Document {str(i)}\n" + doc for i, doc in enumerate(retrieved_docs_text)])

    # Get conversation history
    chat_history = "\n".join([
        f"user: {msg.content}" if isinstance(msg, HumanMessage) else f"assistant: {msg.content}"
        for msg in message_history.messages
    ]) or "No previous conversation."

    if retrieved_docs_text:
        answer = gpt_generate(chat_history, query, prompt_rag_context, context) # context[:CHUNK_SIZE] provare
        # answer = gpt_generate( query, prompt_rag_context, context) 
        # Genera il testo delle fonti con link cliccabili, ma senza mostrare l'URL completo
        sources_text = "\n".join([
            f"🔹 [{doc.metadata['source']}, Pagina {doc.metadata['page']}]"
            f"(http://localhost:8080/viewer.html?file={doc.metadata['source']}&page={doc.metadata['page']}&highlight={quote(doc.page_content[:CHUNK_SIZE])})"
            for doc in docs
        ])
        
        answer = answer + "\n\n" + sources_text

    else:
        answer = gpt_generate(chat_history, query, prompt_rag_no_context, context) 
        #answer = gpt_generate( query, prompt_rag_no_context, context) 

    # Update message history
    message_history.add_user_message(query)
    message_history.add_ai_message(answer)

    return [{"role": "assistant", "content": answer}]

        

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
