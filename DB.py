import psycopg2
import sys
from psycopg2 import sql

def connect_db(host, database, user,password,port):

    try:
        #Connessione a PostgreSQL
        conn = psycopg2.connect(
            
            host= host,
            database= database,
            user= user,
            password= password,
            port= port,
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
    query = sql.SQL("""
        INSERT INTO {} (content, source, embedding, page_number, language)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
    """).format(sql.Identifier(table_name))  # Formattazione sicura per il nome della tabella

    cursor.execute(query, (text, source, embedding_vector, page_number, detected_language))


def drop_index(cursor):
    cursor.execute('DROP INDEX IF EXISTS embeddings_hnsw_index')

def drop_table(cursor,table_name):
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    drop_index(cursor)

def create_table(cursor,table_name):
    cursor.execute(f"""CREATE TABLE {table_name} ( id SERIAL PRIMARY KEY,content TEXT, embedding VECTOR(1024), source TEXT, page_number INT, language TEXT, tsv_content tsvector DEFAULT NULL)""" )
           

def set_tsv(cursor, detected_language,record_id,table_name):
    query = sql.SQL("""
        UPDATE {}  
        SET tsv_content = to_tsvector(
            %s, unaccent(content)
        )
        WHERE id = %s;
    """).format(sql.Identifier(table_name))  
    cursor.execute(query, (detected_language,record_id))   