import DB
import search_v2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from openai import OpenAI
import os
from dotenv import load_dotenv
from urllib.parse import quote
from load_dataset import load_dataset
from evaluate import load
import pandas as pd
from tqdm import tqdm  

bertscore = load("bertscore")
bleu = load("bleu")
rouge = load("rouge")

# Load environment variables from .env file
load_dotenv(override=True)

# Retrieve your API key from your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

MAX_NEW_TOKENS = 4096  # Limita la generazione del modello  
MAX_LENGTH = 4096   #128000 maximum sequence length for the model
CHUNK_SIZE=512



# RAG Prompt Template with context
prompt_rag_context = [
    {"role": "developer", "content": "You are an AI assistant specialized in answering questions based on the provided context.\nFollow these guidelines:\n- Use only the given context to generate your answer.\n- Provide a clear and relevant response. Avoid unnecessary details.\n- Do NOT mention the context, sources, or any document references in your response."},
    {"role": "user", "content": "Here is the relevant context:\n{context}\n---\nNow, answer the following question based strictly on the context.\n\nQuestion: {question}"},
]
# RAG Prompt Template no context
prompt_rag_no_context = [
    {"role": "developer", "content": "You are an AI assistant specialized in answering questions based on the past history.Provide a clear and relevant response. Avoid unnecessary details.\n"},
    {"role": "user", "content": "Now, answer the following question based on your konwladge .\n\nQuestion: {question}"}
]


def salva_risultati_metriche(questions, answers_reference, answers_prediction,sources_prediction,bert_results, bleu_results, rouge_results, nome_file="risultati_metriche_character_splitter.xlsx"): #cambia il nome del file a tuo piacimento
    df = pd.DataFrame({
        "query": questions,
        "answer_reference": answers_reference,
        "answer_prediction": answers_prediction,
        "sources_prediction":sources_prediction,
        "bert_precision": bert_results["precision"],
        "bert_recall": bert_results["recall"],
        "bert_f1": bert_results["f1"],
        "bleu": [results_bleu["bleu"]] * len(questions),  # stesso punteggio per tutte le righe
        "rouge1_f1": rouge_results["rouge1"],
        "rouge2_f1": rouge_results["rouge2"],
        "rougeL_f1": rouge_results["rougeL"]
        
    })

    path_output = os.path.join(os.getcwd(), nome_file)
    df.to_excel(path_output, index=False)
    print(f"\nRisultati salvati in: {path_output}")



def gpt_generate(question, prompt, context=""):

    formatted_prompt = [
        {"role": msg["role"], "content": msg["content"].format(context=context, question=question)}
        for msg in prompt
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=formatted_prompt
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
    results = search_v2.hybrid_search(cursor, query, query_embedding_str)

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
def answer_prediction(query):

    docs = retrieve_documents(query)
    retrieved_docs_text = [doc.page_content for doc in docs]
    context = "\nExtracted documents:\n" + "".join([f"Document {str(i)}\n" + doc for i, doc in enumerate(retrieved_docs_text)])

    if retrieved_docs_text:
        answer = gpt_generate(query, prompt_rag_context, context)
        # Genera il testo delle fonti con link cliccabili, ma senza mostrare l'URL completo
        sources_text = "\n".join([
            f"[{doc.metadata['source']}, Pagina {doc.metadata['page']}]"
            for doc in docs
        ])
        
    else:
        answer = gpt_generate(query, prompt_rag_no_context, context)
        sources_text='null'

    return answer,sources_text


        

if __name__ == "__main__":
        
    # Load database connection
    cursor, conn = DB.connect_db(
        host = "localhost",
        database = "rag_db",
        user = "rag_user",
        password = "password",
        port = "5432"
    )
    print("DEBUG--DataBase connected\n")
    # Load dataset
    questions, answers_reference = load_dataset('.\documenti\dataset\domande_risposte.xlsx') # Sostituisci con il nome del tuo dataset file
    print("DEBUG--Loaded dataset\n")
    # Load embeddings model
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # List of answers generated
    answers_prediction=[]
    # List of sources
    sources_prediction=[]


    for i in tqdm(range(len(questions)), desc="Elaborazione delle domande", unit="domanda"):
        answer,sources_text = answer_prediction(questions[i])
        answers_prediction.append(answer)
        sources_prediction.append(sources_text)


    # BERTScore
    results_bert = bertscore.compute(
        predictions=answers_prediction,
        references=answers_reference,
        model_type="distilbert-base-uncased"
    )

    # ROUGE
    rouge = load("rouge")
    results_rouge = rouge.compute(
        predictions=answers_prediction,
        references=answers_reference
    )

    # BLEU
    bleu = load("bleu")
    # Wrappiamo ogni stringa in una lista
    answers_reference_bleu = [[ref] for ref in answers_reference]

    results_bleu = bleu.compute(
        predictions=answers_prediction,
        references=answers_reference_bleu
    )


    salva_risultati_metriche(
        questions=questions,
        answers_reference=answers_reference,
        answers_prediction=answers_prediction,
        sources_prediction=sources_prediction,
        bert_results=results_bert,
        bleu_results=results_bleu,
        rouge_results=results_rouge
    )

