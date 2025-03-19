from langchain_huggingface import HuggingFaceEmbeddings
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect
from sentence_transformers import CrossEncoder

# Carica il modello di cross-encoder per assegnare un nuovo punteggio ai risultati. Prenderà in input la query e ogni risultato testuale, e calcolerà un punteggio di rilevanza ordinando i risultati
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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

def get_text (elem):
    return str(elem[1])

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

    """
    Trova i chunks con uno score hybrido
    """

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
            ts_rank_cd(tsv_content, to_tsquery(%s, %s) , 0) AS rank_bm25
        FROM embeddings
        WHERE tsv_content @@ to_tsquery(%s, %s)
        ORDER BY rank_bm25 DESC
        LIMIT 100
    ),
    semantic_results AS (
        SELECT source, content, page_number, 
            1 - (embedding <=> %s) / 2  AS rank_semantic
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
    WHERE (0.3 * COALESCE(b.rank_bm25, 0) + 1 * COALESCE(s.rank_semantic, 0)) > 0.92
    ORDER BY final_rank DESC
    LIMIT 20;
    """
    # Esecuzione della query
    cursor.execute(query_sql, (detected_language,query,detected_language, query, query_embedding_str))

    # Recupero dei risultati
    results = cursor.fetchall()

    # Applica il reranking
    reranked_results = rerank_results(query, results)

    return reranked_results



