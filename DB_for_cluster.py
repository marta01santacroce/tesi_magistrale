import psycopg2
import sys
from psycopg2 import sql
import re
import hashlib
import os
import re

def clean_filename_like_gradio(filename):
    
    name, ext = os.path.splitext(filename)
    # Rimuove caratteri indesiderati
    name = name.replace(",", "")                     # virgole 
    name = re.sub(r"[\[\]\(\)]", "", name)            # rimuove parentesi tonde e quadre
    name = re.sub(r"\s+", " ", name).strip()          # rimuove spazi multipli
    return name + ext
 

def hash_file_name(filename):
    # Calcola l'hash SHA256 del nome del file
    hash_object = hashlib.sha256(filename.encode('utf-8'))
    
    # Converte l'hash in un intero
    hash_int = int(hash_object.hexdigest(), 16)
    return hash_int

def check_esistenza_file(source,cursor,table_name ):

    source_formatted = clean_filename_like_gradio(source)
    hash_value = str(hash_file_name(source_formatted))

    # Verifica se esiste già un record con lo stesso hash_value
    check_query = sql.SQL("SELECT 1 FROM {} WHERE hash_value = %s LIMIT 1").format(sql.Identifier(table_name))

    cursor.execute(check_query, (hash_value,))
    if cursor.fetchone():
        return None
    else: 
        return 0


def connect_db(host, database, user,password,port):

    try:
        #Connessione a PostgreSQL
        conn = psycopg2.connect(
            host = host,
            database = database,
            user = user,
            password = password,
            port = port,
        )
    except Exception as error:
        print("An error occurred:", type(error).__name__, "–", error) 
        print("ERROR: open and activate docker container 'postgres_pgvector' ")
        sys.exit(1) 

    
    cursor = conn.cursor()
    cursor.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
    cursor.execute('CREATE EXTENSION IF NOT EXISTS unaccent')
    conn.commit()
    return cursor, conn    

def close_db(cursor, conn):
    # Chiude la connessione al database
    cursor.close()
    conn.close()

def save_changes(conn):
    #Salva le modifiche
    conn.commit()


def insert_embedding(text, source, embedding_vector, page_number, detected_language, cursor,table_name):

    source_formatted = clean_filename_like_gradio(source)
    hash_value = str(hash_file_name(source_formatted))
    
    query = sql.SQL("""
        INSERT INTO {} (content, source, embedding, page_number, language, hash_value)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
    """).format(sql.Identifier(table_name)) 

    cursor.execute(query, (text, source_formatted, embedding_vector, page_number, detected_language, hash_value))


def drop_index(cursor):
    cursor.execute('DROP INDEX IF EXISTS embeddings_hnsw_index')

def drop_table(cursor,table_name):
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    drop_index(cursor)

def create_table(cursor,table_name):
    cursor.execute(f"""CREATE TABLE {table_name} ( id SERIAL PRIMARY KEY,content TEXT, embedding VECTOR(1024), source TEXT, page_number INT, language TEXT, tsv_content tsvector DEFAULT NULL, hash_value TEXT, cluster_id TEXT DEFAULT '-1', cluster_embedding VECTOR(1024),combined_embedding vector (1024))""" )
           
def set_tsv(cursor, detected_language,record_id,table_name):
    query = sql.SQL("""
        UPDATE {}  
        SET tsv_content = to_tsvector(
            %s, unaccent(content)
        )
        WHERE id = %s;
    """).format(sql.Identifier(table_name))  
    cursor.execute(query, (detected_language,record_id)) 


