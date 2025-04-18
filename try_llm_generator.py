import DB
import search_v2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


READER_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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
    max_new_tokens = 1000,
)


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
        "content": """Here is the relevant context:  
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
    results = search_v2.hybrid_search(cursor, query, query_embedding_str)

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

    cursor, conn = DB.connect_db()
    # Modello per embeddings
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

        # sorgenti + pagina
        sources_with_pages = {(doc.metadata["source"], doc.metadata["page"]) for doc in docs}



        retrieved_docs_text = [doc.page_content for doc in docs]  # We only need the text of the documents

        context = "\nExtracted documents:\n"

        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

  
        if context:
            final_prompt = RAG_PROMPT_TEMPLATE.format(question=query, context=context)

            # Genera la risposta
            answer = READER_LLM(final_prompt)[0]["generated_text"]
            print("\n" + "="*150)
            print("\nRisposta Generata:\n")
            print(answer)

            print("\nFonti utilizzate:\n")
            for source, page in sources_with_pages:
                print(f"🔹 Fonte: {source}, Pagina: {page}")


        else:
            print("\nNo results found.")
 
    # Chiude la connessione al database
    DB.close_db(cursor, conn)
