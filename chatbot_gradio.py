
import gradio as gr  #versione 5.20.1
import DB
import search_v2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


READER_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Load Model
MAX_NEW_TOKENS = 1000  # Limita la generazione del modello
MAX_LENGTH = 2048 # maximum sequence length for the model

tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)

# Create LLM pipeline
READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True, 
    temperature=0.2,
    repetition_penalty = 1.1,
    return_full_text=False,
    max_new_tokens=MAX_NEW_TOKENS,
)


# RAG Prompt Template
prompt_rag = [
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
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(prompt_rag, tokenize=False, add_generation_prompt=True)


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


# Chatbot function
def chatbot_response(query,history):

    """Gestisce la risposta del chatbot. Ignora input vuoti."""
    if not query.strip():  # Evita invii di stringhe vuote
       return [{"role": "assistant", "content": "⚠️ Empty user message"}]

    else:
        docs = retrieve_documents(query)
        sources_with_pages = {(doc.metadata["source"], doc.metadata["page"]) for doc in docs}

        retrieved_docs_text = [doc.page_content for doc in docs]
        context = "\nExtracted documents:\n" + "".join([f"Document {str(i)}\n" + doc for i, doc in enumerate(retrieved_docs_text)])

    
        if retrieved_docs_text:

            final_prompt = RAG_PROMPT_TEMPLATE.format(question=query, context=context[:MAX_LENGTH])

            answer = READER_LLM(final_prompt)[0]["generated_text"]

            sources_text = "\n".join([f"🔹 {source}, Page {page}" for source, page in sources_with_pages])
            
            output = answer + "\n\n" + sources_text

            #print("DEBUG----output\n" + str(output))

        else:

            # Costruisce dinamicamente il messaggio con la query
            prompt_llm = [
                {
                    "role": "system",
                    "content": "You are a friendly chatbot and an assistant for question-answering tasks. you must answer the user query.",
                },
                {   
                    "role": "user", 
                    "content": f"Query: {query}"
                },  
            ]

            # Genera il prompt con il messaggio aggiornato
            LLM_PROMPT_TEMPLATE = READER_LLM.tokenizer.apply_chat_template(prompt_llm, tokenize=False, add_generation_prompt=True)

            # Genera la risposta
            output = READER_LLM(LLM_PROMPT_TEMPLATE, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.5, top_k=50, top_p=0.95)
            #print("DEBUG--llm: \n")
            #print(output)
        
        return [{"role": "assistant", "content": output}]

# Gradio Interface
chatbot_ui = gr.ChatInterface(

    fn = chatbot_response,
    type = 'messages',
    title = "🕷️RAG Chatbot🕷️",
    theme ='allenai/gradio-theme',
    show_progress ='full',
    fill_height =True,
    save_history =True,
    flagging_mode ="manual",

)

if __name__ == "__main__":
        
    # Load database connection
    cursor, conn = DB.connect_db()
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    
    chatbot_ui.launch()
