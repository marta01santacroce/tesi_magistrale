import os
import pymupdf
import sys
from pathlib import Path
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import DB
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

TABLE_NAME = 'embeddings_semantic_splitter_percentile'

# Dizionario con sigle delle lingue e i nomi delle lingue supportate
language_dict = {
    "ar": "Arabic", "ca": "Catalan", "da": "Danish", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "hi": "Hindi", "hu": "Hungarian",
    "id": "Indonesian", "it": "Italian", "lt": "Lithuanian", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sv": "Swedish", "ta": "Tamil", "tr": "Turkish"
}

BATCH_SIZE = 50

# Funzione per rilevare la lingua del testo
def detect_language(text):
    try:
        lang = detect(text)
        return language_dict.get(lang, "simple")
    except LangDetectException:
        return "simple"

# Funzione per pulire il testo
def clean_text(text):
    return text.replace('\x00', '').replace('\n', ' ').replace('\t', ' ')

# Funzione per estrarre testo dalle pagine PDF
def process_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text_list = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if text.strip():
            text_list.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(pdf_path), "page_number": page_num + 1}
            ))
    return text_list

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("[ERROR] Pass two parameters: \n- the path to the file or folder \n- Y to reset DB, N to keep it")
        sys.exit(1)

    start_time = time.time()

    cursor, conn = DB.connect_db(
        host="localhost",
        database="rag_db",
        user="rag_user",
        password="password",
        port="5432"
    )

    if sys.argv[2] == 'Y':
        DB.drop_table(cursor, table_name=TABLE_NAME)
        DB.create_table(cursor, table_name=TABLE_NAME)

    path = Path(sys.argv[1])

    if path.is_file():
        print(f"\n{path} is a file.")
        pdf_files = [path]
    elif path.is_dir():
        print(f"\n{path} is a directory.")
        pdf_files = [path / f for f in os.listdir(path) if f.endswith(".pdf")]
    else:
        print(f"{path} does not exist.")
        sys.exit(1)

    # Modello leggero per SemanticChunker (velocizza drasticamente lo split)
    splitter_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

    # Modello pesante per l'embedding finale (alta qualità e multilingua)
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    text_splitter = SemanticChunker(
        embeddings=splitter_model,
        breakpoint_threshold_type="percentile"
    )

    all_documents = []
    for i, pdf in enumerate(pdf_files):
        print(f"[{i+1}/{len(pdf_files)}] Processing: {pdf.name}")
        docs = process_pdf(pdf)
        all_documents.extend(docs)

    print(f"\nSplitting {len(all_documents)} documents")
    chunks = []
    for i in tqdm(range(0, len(all_documents), 10), desc="Splitting in batches"):
        batch_docs = all_documents[i:i+10]
        batch_chunks = text_splitter.split_documents(batch_docs)
        chunks.extend([chunk for chunk in batch_chunks if chunk.page_content.strip()])

    print(f"\n{len(chunks)} chunks generated. Starting embedding and DB insert\n")

    batch = []
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks"), 1):
        text = clean_text(chunk.page_content)
        formatted_text = f"passage: {text}"
        source = chunk.metadata.get("source", "")
        page_number = chunk.metadata.get("page_number", 0)

        detected_language = detect_language(text)
        embedding_vector = embedding_model.embed_query(formatted_text)

        batch.append((text, source, embedding_vector, page_number, detected_language))

        if len(batch) == BATCH_SIZE or i == len(chunks):
            for entry in batch:
                DB.insert_embedding(*entry, cursor, table_name=TABLE_NAME)
                record_id = cursor.fetchone()[0]
                DB.set_tsv(cursor, entry[4], record_id, table_name=TABLE_NAME)
            DB.save_changes(conn)
            #print(f"\nInserted {i}/{len(chunks)} chunks")
            batch.clear()

    DB.close_db(cursor, conn)
    print(f"\nDone. {len(chunks)} chunks saved. Total time: {round(time.time() - start_time, 2)}s")
