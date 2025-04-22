# RAG system for master thesis
## *Ingegneria Informatica Data Engineering and Analytics (UNIMORE)*

File:
- chatbot_gradio.py ->  code for the chatbot realised by gradio with OpenAI (gpt-4o-mini LLM)
- save_embeddings_v_2.py -> code for saving documents in PGVector DB
- DB.py -> code to handle all interactions with the DB
- search_v2.py -> code to retrieve relevant documents with hybrid search + reranking
- evaluation.py -> code for evaluate the rag sysem using some metrics (BERT, BLEU and ROUGE)
- load_dataset.py -> code for load dataset (questions/answers)
- **the other .py files are not rilevant (They are only used for personal debugging)**
