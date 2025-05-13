import DB
import search_v2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage

READER_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 1000  # Limita la generazione del modello
MAX_LENGTH = 2048 # maximum sequence length for the model

TABLE_NAME = 'embeddings' # nome della tabella da cui effettuare la ricerca ibrida -> 
# CAMBIARE IL NOME DELLA TABELLA DI INTERESSE!!!!! -> 
    # embeddings -> usa recursive character splitter
    # embeddings_character_splitter -> usa character splitter
    # embeddings_semantic_splitter_percentile -> usa semantic splitter con percentile come breakpoint_threshold_type


    #NON ANCORA IMPLEMENTATE
    # embeddings_semantic_splitter_standard_deviation -> usa semantic splitter con tandard_deviation come breakpoint_threshold_type
    # embeddings_semantic_splitter_interquartile -> usa semantic splitter con interquartile come breakpoint_threshold_type
    # embeddings_semantic_splitter_gradient -> usa semantic splitter con gradient  come breakpoint_threshold_type



tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)


READER_LLM = pipeline(
    model = model,
    tokenizer = tokenizer,
    task = "text-generation",
    do_sample = True,
    temperature = 0.2,
    repetition_penalty = 1.1,
    return_full_text = False,
    max_new_tokens = MAX_NEW_TOKENS,
)


# Inizializza la memoria della conversazione
message_history = ChatMessageHistory()

# Prompt con memoria
prompt_in_chat_format  = [
    {
        "role": "system",
        "content": """You are an AI assistant specialized in answering questions based on the provided context.  
        Follow these guidelines:  
        - Use only the given context to generate your answer. 
        - Provide a clear, concise, and relevant response. Avoid unnecessary details.  
        - Do NOT mention the context, sources, or any document references in your response.    
        """,
    },
    {
        "role": "user",
        "content": """Previous conversation:  
        {chat_history}  
        ---  
        Here is the relevant context:  
        {context}  
        ---  
        Now, answer the following question based strictly on the context.  

        Question: {question}""",
    },
]

RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)



def retrieve_documents(query):
    """Recupera i documenti più rilevanti dal database basandosi sulla query dell'utente."""

    # Genera l'embedding della query
    formatted_query = f"query: {query}"
    query_embedding = embedding_model.embed_query(formatted_query)
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    # Esegue la ricerca nel database (BM25 + Similarità Coseno + reranking)
    results = search_v2.hybrid_search(cursor, query, query_embedding_str, table_name=TABLE_NAME)

    # Converte i risultati in oggetti Document di LangChain
    documents = [
        Document(
            page_content=res[1],
            metadata={
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

if __name__ == "__main__":

    cursor, conn = DB.connect_db(
        host = "localhost",
        database = "rag_db",
        user = "rag_user",
        password = "password",
        port = "5432"
    )
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    print("\nHybrid Search (BM25 + Cosine Similarity)")

    while True:
        # Query dinamica dall'utente
        query = input("\nEnter your question or press enter without text to exit: ")

        # Se l'utente non inserisce nulla, esce
        if not query.strip():
            print("No query entered. Exiting...")
            break


        # Recupera i documenti pertinenti
        docs = retrieve_documents(query)
        sources_with_pages = {(doc.metadata["source"], doc.metadata["page"]) for doc in docs}
        retrieved_docs_text = [doc.page_content for doc in docs]  # We only need the text of the documents
        context = "\nExtracted documents:\n" + "".join([f"Document {str(i)}\n" + doc for i, doc in enumerate(retrieved_docs_text)])

         # Get conversation history
        chat_history = "\n".join([f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in message_history.messages])
  
        if retrieved_docs_text:
            
            final_prompt = RAG_PROMPT_TEMPLATE.format(question=query, context=context[:MAX_LENGTH],chat_history=chat_history)

            # Genera la risposta
            answer = READER_LLM(final_prompt)[0]["generated_text"]
            print("\n" + "="*150)
            print("\nRisposta Generata:\n")
            print(answer)

            print("\nFonti utilizzate:\n")
            for source, page in sources_with_pages:
                print(f"🔹 Fonte: {source}, Pagina: {page}")

            # Update message history
            message_history.add_user_message(query)
            message_history.add_ai_message(answer)

        else:
            print("\nNo results found.")
 
    # Chiude la connessione al database
    DB.close_db(cursor, conn)
