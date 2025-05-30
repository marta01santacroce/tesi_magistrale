# RAG System for Master Thesis  
## *Ingegneria Informatica - Data Engineering and Analytics (UNIMORE)*

Questo progetto implementa un sistema RAG (*Retrieval-Augmented Generation*) basato su modelli LLM di OpenAI e un database vettoriale gestito tramite PGVector.

## 🗂️ File principali

- `chatbot_gradio.py`  
  Codice per l'interfaccia del chatbot sviluppata con Gradio e basata su LLM (GPT-4o-mini) di OpenAI.

- `save_embeddings_character.py`  
  Salvataggio dei documenti nel database PGVector usando il metodo di suddivisione per caratteri.

- `save_embeddings_semantic.py`  
  Salvataggio dei documenti nel database PGVector usando uno splitter semantico.

- `save_embeddings_recursive.py`  
  Salvataggio dei documenti nel database PGVector usando uno splitter ricorsivo basato sui caratteri.

- `DB.py`  
  Gestione di tutte le interazioni con il database vettoriale.

- `search_v2.py`  
  Recupero dei documenti rilevanti tramite ricerca ibrida e reranking.

- `evaluation.py`  
  Valutazione del sistema RAG tramite metriche come BERTScore, BLEU, ROUGE, Recall@K e MRR.

- `load_dataset.py`  
  Caricamento del dataset contenente coppie domanda/risposta.

- `update_name_pdf_and_move_folder.py`  
  Gestione dei nuovi file PDF: formatta i nomi come richiesto da Gradio e li copia nella cartella `pdf_files`.

---
## 🛢️ DataBase 
Gestione del container docker con le istanze in PgVector (sezione in fase di modifica)

## 🚀 Avvio del sistema

Per avviare il chatbot, esegui il file:

```bash
python chatbot_gradio.py
```
Questo script utilizza i seguenti moduli fondamentali:
- `DB.py`
  per tutte le operazioni con il database vettoriale (PGVector).
- `save_embeddings_recursive.py`
  per salvare i documenti nella base di conoscenza utilizzando una suddivisione ricorsiva.
- `update_name_pdf_and_move_folder.py`
  per formattare i nomi dei file PDF e copiarli correttamente nella cartella dedicata.

## 📂 Cartella pdf_files
Assicurati che nella directory principale del progetto sia presente una cartella chiamata pdf_files/.
Tutti i documenti PDF utilizzati dal sistema verranno inseriti in questa cartella dopo l'upload tramite interfaccia.
A causa delle dimensioni, la cartella `pdf_files/` con i file correnti nel databse non è presente nel repository.  
Puoi scaricarla da Google Drive al seguente link:
🔗 [Scarica pdf_files da Google Drive](https://drive.google.com/file/d/1Nzn8ZO0bOhIyewmZWZIPRk4Gnsp-zeiw/view?usp=drive_link)

Dopo aver scaricato la cartella, posizionala nella directory principale del progetto.

Il file `update_name_pdf_and_move_folder.py` si occupa di:

- Rinominare i file PDF secondo il formato richiesto da Gradio
- Copiarli automaticamente nella cartella pdf_files/

## ✅ Requisiti
Assicurati di avere installato:

- Python 3.10 (il sistema è stato testato con questa versione e non viene quindi assiucrato il funzionamento con versioni precedenti)
- Le librerie specificate in `requirements.txt`


## ⚙️ Configurazione Variabili d'Ambiente

Per far funzionare correttamente il progetto, è necessario creare un file `.env` nella directory principale e definire:

- la chiave API di OpenAI con la variabile:

```env
OPENAI_API_KEY=la_tua_chiave_openai
```

- i nomi per le variabili di connessione al database:

```env
HOST_NAME = nome_host
DATABASE_NAME = nome_database
USER_NAME = nome_utente
PASSWORD = password
PORT = numero_porta
```
Per caricare le variabili presenti nel file `.env` si usa la parte di codice già presente, ossia:
```python
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)
# Retrieve your API key from your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#parametri  di connessione al DB
HOST_NAME = os.getenv("HOST_NAME")
DATABASE_NAME = os.getenv("DATABASE_NAME")
USER_NAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("PASSWORD")
PORT = os.getenv("PORT")

```

