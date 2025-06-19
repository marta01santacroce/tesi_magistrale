from langdetect import detect
from sentence_transformers import CrossEncoder

# Carica il modello di cross-encoder per assegnare un nuovo punteggio ai risultati. Prenderà in input la query e ogni risultato testuale, e calcolerà un punteggio di rilevanza ordinando i risultati
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

    # Restituisce solo i top k risultati ordinati
    return [res for res, _ in reranked_results[:10]]


def hybrid_search(cursor,query, query_embedding_str,table_name):

    """
    Trova i chunks con uno score semantico sull'embedding combinato
    """
    # Definizione della query con limit 20/1000
    query_sql = f"""
    
        SELECT source, content, page_number,
            1 - (combined_embedding <=> %s) / 2 AS rank_semantic
            
        FROM {table_name}
        
        ORDER BY rank_semantic DESC
        LIMIT 20

    """

    cursor.execute(query_sql, (query_embedding_str,))

    # Recupero dei risultati
    results = cursor.fetchall()

    # Applica il reranking
    reranked_results = rerank_results(query, results)

    return reranked_results



