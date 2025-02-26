import psycopg2
import sys


def connect_db():

    try:
        #Connessione a PostgreSQL
        conn = psycopg2.connect(
            
            host= "localhost",
            database= "rag_db",
            user= "rag_user",
            password= "password",
            port= "5432",
        )
    except:
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


def insert_embedding(text,source, embedding_vector, page_number, detected_language,cursor):
    # Inserisci il documento con la lingua rilevata
    cursor.execute(
        """
        INSERT INTO embeddings (content, source,embedding, page_number, language)
        VALUES (%s, %s, %s, %s,%s)
        RETURNING id;
        """,
        (text,source, embedding_vector, page_number, detected_language)
    )

def drop_index(cursor):
    cursor.execute('DROP INDEX IF EXISTS embeddings_hnsw_index')

def drop_table(cursor):
    cursor.execute('DROP TABLE IF EXISTS embeddings')
    drop_index(cursor)

def create_table_embeddings(cursor):
    cursor.execute('CREATE TABLE embeddings ( id SERIAL PRIMARY KEY,content TEXT, embedding VECTOR(1024), source TEXT, page_number INT, language TEXT, tsv_content tsvector DEFAULT NULL)' )
           

def set_tsv(cursor, detected_language,record_id):
    cursor.execute(
            """
            UPDATE embeddings  
            SET tsv_content = to_tsvector(
               %s, unaccent(content)
            )
            WHERE id = %s;
            """,
            (detected_language,record_id)
        )          