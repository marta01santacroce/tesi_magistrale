from langchain_huggingface import HuggingFaceEmbeddings
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect
import DB
from sentence_transformers import CrossEncoder

# Dizionario con sigle delle lingue e i nomi delle lingue supportate
language_dict = {

    "ar": "Arabic",
    "ca": "Catalan",
    "da": "Danish",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "ta": "Tamil",
    "tr": "Turkish"
}

# Carica il modello di cross-encoder per assegnare un nuovo punteggio ai risultati. Prenderà in input la query e ogni risultato testuale, e calcolerà un punteggio di rilevanza ordinando i risultati
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_source (elem):
    return str(elem[0])

def get_text (elem):
    return str(elem[1])

def get_page (elem):
    return str(elem[2])

def get_bm25_score (elem):
    return str(elem[3])

def get_semantic_score (elem):
    return str(elem[4])

def get_hybrid_score (elem):
    return str(elem[5])

def rerank_results(query, results):
    """
    Reranka i risultati usando un modello Cross-Encoder.
    """
    if not results:
        return []

    # Prepara i pair (query, documento)
    pairs = [(query, get_text(res)) for res in results]

    # Ottiene i punteggi dal modello cross-encoder
    scores = reranker.predict(pairs)

    # Associa i punteggi ai risultati
    reranked_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    # Restituisce i risultati ordinati
    return [res for res, _ in reranked_results]

def hybrid_search(cursor, query,query_embedding_str):

    # Rileva la lingua in automatico
    try:
        detected_language = detect(query)
        detected_language = language_dict.get(detected_language)
        if detected_language is None:
            detected_language = "simple"  # Usa un fallback

    except LangDetectException:
        detected_language = "simple"  # Usa un fallback

    query_terms = query.split() 
    query = " & ".join(query_terms)  
    # Definizione della query combinata
    query_sql = """
    WITH bm25_results AS (
        SELECT source, content, page_number, 
            ts_rank_cd(tsv_content, to_tsquery(%s, %s)) AS rank_bm25
        FROM embeddings
        WHERE tsv_content @@ to_tsquery(%s, %s)
        ORDER BY rank_bm25 DESC
        LIMIT 100
    ),
    semantic_results AS (
        SELECT source, content, page_number, 
            1 - (embedding <=> %s) / 2 AS rank_semantic
        FROM embeddings
        ORDER BY rank_semantic DESC
        LIMIT 100
    )
    SELECT 
        COALESCE(b.source, s.source) AS source,
        COALESCE(b.content, s.content) AS content,
        COALESCE(b.page_number, s.page_number) AS page_number,
        COALESCE(b.rank_bm25, 0) AS rank_bm25,
        COALESCE(s.rank_semantic, 0) AS rank_semantic,
        (0.3 * COALESCE(b.rank_bm25, 0) + 1 * COALESCE(s.rank_semantic, 0)) AS final_rank
    FROM bm25_results b
    FULL OUTER JOIN semantic_results s 
    ON b.source = s.source AND b.content = s.content AND b.page_number = s.page_number
    WHERE (0.3 * COALESCE(b.rank_bm25, 0) + 1 * COALESCE(s.rank_semantic, 0)) > 0.9
    ORDER BY final_rank DESC
    LIMIT 10;
    """
    # Esecuzione della query
    cursor.execute(query_sql, (detected_language,query,detected_language, query, query_embedding_str))

    # Recupero dei risultati
    results = cursor.fetchall()
    return results


if __name__ == "__main__":

    cursor, conn = DB.connect_db()
    # Modello per embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    print("\nHybrid Search (BM25 + Cosine Similarity)")

    #print("\nType your query and press Enter or press Enter without text to exit.")

    while True:
        # Query dinamica dall'utente
        query = input("\nEnter your question or press enter without text to exit: ")

        # Se l'utente non inserisce nulla, esce
        if not query.strip():
            print("No query entered. Exiting...")
            break

        formatted_query = f"query: {query}"    
        query_embedding = embedding_model.embed_query(formatted_query)

        # Converti l'array di embedding in stringa per PostgreSQL
        query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        results = hybrid_search(cursor, query,query_embedding_str)

        # print results con reranking
        print("\nDEBUG---print results con reranking\n")

        if results:
            
            # Applica il reranking
            reranked_results = rerank_results(query, results)

            for indice, elem in enumerate(reranked_results):
                
                indice += 1
                print("\n" + "="*150)
                print("\n"+str(indice)+")\n")
                print("\n-SOURCE: " + get_source(elem))
                print("\n-TEXT: " + get_text(elem))
                print("\n-PAGE_NUMBER: " + get_page(elem))
                print("\n-BM_25_SCORE: " + get_bm25_score(elem))
                print("\n-SEMANTIC_SCORE: " + get_semantic_score(elem))
                print("\n-HYBRID_SCORE (0.3 * BM_25_SCORE + 1 * SEMANTIC_SCORE): " + get_hybrid_score(elem))

        else:
            print("\nNo results found.")
 
    # Chiude la connessione al database
    DB.close_db(cursor, conn)

