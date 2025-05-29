import os
import pymupdf
import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import DB
from tqdm import tqdm

CHUNK_SIZE = 512
CHUNK_OVERLAP = 10
TABLE_NAME = 'embeddings_recursive'

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
def save_pdfs(file_path, option,cursor,table_name,embedding_model,conn):
    is_file = False
    is_dir = False
       
    path = Path(file_path)

    # Check if it's a file or a directory
    if path.is_file():

        #print(f"\n{path} is a file.\n")
        is_file = True
        pdf_files = path

        source = os.path.basename(path)
        value = DB.check_esistenza_file(source, cursor, table_name=table_name)

        if value is None: 
            print(f"[ATTENZIONE PROCESSO TERMINATO] Il file '{pdf_files}' è già presente nel DB. Non sono ammessi duplicati\n")
            return None

    elif path.is_dir():

        #print(f"\n{path} is a directory.")
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

    if option == 'Y':

        DB.drop_table(cursor,table_name=table_name)
        DB.create_table(cursor,table_name=table_name)
             

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

        #Salvare embeddings
        for chunk in tqdm(chunks, desc="Elaborazione chunk"):

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
        
        print(f"{len(chunks)} chunk salvati nel database con embeddings.")

    return value

if __name__ == "__main__":

    cursor, conn = DB.connect_db( 
        host = "localhost",
        database = "rag_db",
        user = "rag_user",
        password = "password",
        port = "5432")
    
    # Modello per embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    if len(sys.argv) != 3:
        print("[ERROR] Pass two parameters: \n- the path to the individual document file or to the folder containing all the files \n- Y if you want to clear all DB or N if you don't want to do that")
        sys.exit()

    save_pdfs(file_path=sys.argv[1], option=sys.argv[2], cursor=cursor, table_name = TABLE_NAME, embedding_model=embedding_model)



    




