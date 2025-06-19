import os
from dotenv import load_dotenv
import pymupdf
import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import DB_for_cluster as DB
from tqdm import tqdm
import hdbscan
import json
import numpy as np
from collections import defaultdict
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHUNK_SIZE = 512
CHUNK_OVERLAP = 10
TABLE_NAME = 'your_table_name'

# Load environment variables from .env file
load_dotenv(override=True)

#parametri  di connessione al DB
HOST_NAME = os.getenv("HOST_NAME")
DATABASE_NAME = os.getenv("DATABASE_NAME")
USER_NAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("PASSWORD")
PORT = os.getenv("PORT")

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
  
''' 
Parametri da passare in fase di esecuzione:

    1) il path della cartella contenente i file pdf da memorizzare OPPURE il path del singolo file pdf da memorizzare

    2) 'Y' se si vuole inizializzare il DB OPPURE 'N' se non si vuole inizializzare il DB

'''
def fetch_embeddings_by_source(cursor, table_name, source):
    cursor.execute(f"""
        SELECT id, embedding, hash_value 
        FROM {table_name} 
        WHERE source = %s
    """, (source,))
    
    rows = cursor.fetchall()
    ids, embeddings = [], []
    hash_value = None

    for _id, embedding_raw, hv in rows:
        if hash_value is None:
            hash_value = hv  # Assumiamo che sia uguale per tutte le righe

        ids.append(_id)

        if isinstance(embedding_raw, str):
            embedding = np.array(json.loads(embedding_raw), dtype=np.float32)
        elif isinstance(embedding_raw, (bytes, bytearray)):
            embedding = np.frombuffer(embedding_raw, dtype=np.float32)
        else:
            embedding = np.array(embedding_raw, dtype=np.float32)

        if embedding.shape[0] != 1024:
            raise ValueError(f"Embedding size mismatch for id={_id}: expected 1024, got {embedding.shape[0]}")
        embeddings.append(embedding)

    return ids, np.vstack(embeddings), hash_value

def cluster_embeddings(embeddings):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    return clusterer.fit_predict(embeddings)

def update_cluster_ids(cursor, table_name, ids, labels, hash_value):
    values = ', '.join(
        cursor.mogrify("(%s, %s)", (id_, f"{hash_value}_{int(label)}")).decode("utf-8")
        for id_, label in zip(ids, labels)
    )

    query = f"""
    UPDATE {table_name} AS t SET cluster_id = v.cluster_id
    FROM (VALUES {values}) AS v(id, cluster_id)
    WHERE t.id = v.id;
    """
    cursor.execute(query)


def compute_and_update_cluster_embeddings(cursor, table_name, source):
    
    cursor.execute(f"""
    SELECT cluster_id, embedding
    FROM {table_name}
    WHERE split_part(cluster_id, '_', 2) != '-1'
    AND source = %s
    """, (source,))
    
    rows = cursor.fetchall()
    cluster_map = {}

    for cluster_id, embedding_raw in rows:
        if isinstance(embedding_raw, str):
            embedding = np.array(json.loads(embedding_raw), dtype=np.float32)
        elif isinstance(embedding_raw, (bytes, bytearray)):
            embedding = np.frombuffer(embedding_raw, dtype=np.float32)
        else:
            embedding = np.array(embedding_raw, dtype=np.float32)

        if embedding.shape[0] != 1024:
            raise ValueError(f"Embedding size mismatch in cluster {cluster_id}")
        cluster_map.setdefault(cluster_id, []).append(embedding)

    for cluster_id, embeddings_list in cluster_map.items():
        centroid = np.mean(np.vstack(embeddings_list), axis=0)
        cursor.execute(f"""
            UPDATE {table_name}
            SET cluster_embedding = %s
            WHERE cluster_id = %s AND source = %s
        """, (centroid.tolist(), cluster_id, source))

def update_combined_embedding(cursor, table_name):
    # Query
    cursor.execute(f"""
        SELECT id, embedding, cluster_embedding
        FROM {table_name}
        WHERE embedding IS NOT NULL
    """)
    rows = cursor.fetchall()

    for row in rows:
        row_id, embedding_raw, cluster_embedding_raw = row

        # Conversione da stringa JSON a vettore numpy
        try:
            embedding = np.array(json.loads(embedding_raw), dtype=np.float32)
        except Exception as e:
            print(f"Errore parsing embedding per id {row_id}: {e}")
            continue

        if cluster_embedding_raw is not None:
            try:
                cluster_embedding = np.array(json.loads(cluster_embedding_raw), dtype=np.float32)
                combined = embedding * 0.7 + cluster_embedding * 0.3
            except Exception as e:
                print(f"Errore parsing cluster_embedding per id {row_id}: {e}")
                combined = embedding
        else:
            combined = embedding

        # Salva nel DB
        cursor.execute(
            f"UPDATE {table_name} SET combined_embedding = %s WHERE id = %s",
            (combined.tolist(), row_id)
        )
        

def save_pdfs(file_path,cursor,table_name,embedding_model, conn):
    is_file = False
    is_dir = False
       
    path = Path(file_path)

    # Check if it's a file or a directory
    if path.is_file():

        is_file = True
        pdf_files = path

        source = os.path.basename(path)
        value = DB.check_esistenza_file(source, cursor, table_name=table_name)

        if value is None: 
            print(f"[ATTENZIONE PROCESSO TERMINATO] Il file '{pdf_files}' è già presente nel DB. Non sono ammessi duplicati\n")
            return None

    elif path.is_dir():

        is_dir = True
        pdf_folder = path
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

        check_list = pdf_files[:] 
        
        for name in check_list:
            
            source = os.path.basename(name)
            value = DB.check_esistenza_file(source, cursor, table_name=table_name)
                
            if value is None: 
                print(f"[ATTENZIONE] Il file {name} è già presente nel DB. Controlla la tua cartella dei files")
                pdf_files.remove(name)  # Rimuove solo dalla lista
                value = True

    else:
        print(f"{path} does not exist.")

    
    all_documents = []

    if is_file is True and value != None:

        doc =  pymupdf.open(pdf_files)

        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            if text.strip():
                all_documents.append(Document(
                    page_content=text,
                    metadata={"source": os.path.basename(path), "page_number": page_num + 1}
                ))

    elif is_dir is True and len(pdf_files) > 0 and value != None:

        for pdf_file in pdf_files:
             
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc =  pymupdf.open(pdf_path)
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text.strip():
                    all_documents.append(Document(
                        page_content=text,
                        metadata={"source": os.path.basename(pdf_path), "page_number": page_num + 1}
                    ))

    if len(all_documents) > 0:
        #Dividere in chunk
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(all_documents)

        # Raggruppa i chunk per 'source'
        chunks_by_source = defaultdict(list)
        for chunk in chunks:
            source = chunk.metadata["source"]
            chunks_by_source[source].append(chunk)


        # Itera per ogni source (PDF) e processa i chunk associati
        for source, source_chunks in tqdm(chunks_by_source.items(), desc="Elaborazione per file"):
            
            for chunk in source_chunks:

                text = chunk.page_content
                source = chunk.metadata.get("source", "")  # Usa "" come valore predefinito se non esiste
                page_number = chunk.metadata.get("page_number", "")  

                if not isinstance(text, str):
                    text = str(text)  # Converte in stringa se necessario

                if not isinstance(source, str):
                    source = str(source)  # Converte in stringa se necessario

                text = text.replace('\x00', '')
                text = text.replace('\n', ' ')
                text = text.replace('\t', ' ')
                formatted_text = f"passage: {text}"
                source = source.replace('\x00', '')
                
                # Rileva la lingua in automatico
                try:
                    detected_language = detect(text)
                    detected_language = language_dict.get(detected_language)
                    if detected_language is None:
                        detected_language = "simple"  # Usa un fallback

                except LangDetectException:
                    detected_language = "simple"  # Usa un fallback

                # Genera embedding
                embedding_vector = embedding_model.embed_query(formatted_text)          
                DB.insert_embedding(text,source, embedding_vector, page_number, detected_language,cursor,table_name=table_name)
                # Ottieni l'id del record appena inserito
                record_id = cursor.fetchone()[0]
                # Aggiorna il campo tsv_content solo per il record appena inserito
                DB.set_tsv(cursor,detected_language,record_id,table_name=table_name)
                conn.commit()


            logger.info(f"Clustering per documento: {source}")

            source = DB.clean_filename_like_gradio(source)

            ids, embeddings, hash_value = fetch_embeddings_by_source(cursor, table_name, source)
            if len(embeddings) < 2:
                logger.warning(f"Numero di embeddings {len(embeddings)} < 2 troppo basso per clustering nel documento '{source}'. Saltato.")
                continue

            labels = cluster_embeddings(embeddings)

            if set(labels) == {-1}:
                logger.warning(f"Tutti embeddings in '{source}' sono rumore.")
                continue

            update_cluster_ids(cursor, table_name, ids, labels, hash_value)
            conn.commit()
            compute_and_update_cluster_embeddings(cursor, table_name, source)
            conn.commit()
            logger.info(f"Completato clustering per '{source}'")    
        
    
    
    update_combined_embedding(cursor, table_name=TABLE_NAME)
    print(f"{len(chunks)} chunk salvati nel database con embeddings.")

    return value

if __name__ == "__main__":

    # Load database connection
    cursor, conn = DB.connect_db(
        host = HOST_NAME,
        database = DATABASE_NAME,
        user = USER_NAME,
        password = PASSWORD,
        port = PORT
    )
    
    # Modello per embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    if len(sys.argv) != 3:
        print("[ERROR] Pass two parameters: \n- the path to the individual document file or to the folder containing all the files \n- Y if you want to clear all DB or N if you don't want to do that")
        sys.exit()

    option=sys.argv[2]   

    if option == 'Y':
        DB.drop_table(cursor,table_name=TABLE_NAME)
        DB.create_table(cursor,table_name=TABLE_NAME)
             

    save_pdfs(file_path=sys.argv[1], cursor=cursor, table_name = TABLE_NAME, embedding_model=embedding_model, conn=conn)
    # Commit
    conn.commit()
    cursor.close()
    conn.close()



    




