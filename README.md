# RAG system for master thesis
## *Ingegneria Informatica Data Engineering and Analytics (UNIMORE)*

File:
- chatbot_gradio.py ->  code for the chatbot realised by gradio with OpenAI (gpt-4o-mini LLM)
- save_embeddings_character.py -> code for saving documents in PGVector DB with character text splitter
- save_embeddings_semantic.py -> code for saving documents in PGVector DB with semantic text splitter
- save_embeddings_recursive.py -> code for saving documents in PGVector DB with recursive character text splitter
- DB.py -> code to handle all interactions with the DB
- search_v2.py -> code to retrieve relevant documents with hybrid search + reranking
- evaluation.py -> code for evaluate the rag sysem using some metrics (BERT, BLEU, ROUGE, RECALL@K and MRR)
- load_dataset.py -> code for load dataset (questions/answers)
- update_name_pdf_and_move_folder.py -> code to handle the loading of new files, formatting the name as Gradio wishes, and copying the document into the *pdf_files* folder, which MUST be present in the project directory and thus contain all the pdfs
