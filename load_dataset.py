import pandas as pd

def load_dataset(file_path):
    # Percorso del file Excel
    file_path = file_path  

    # Caricamento del file
    df = pd.read_excel(file_path)

    # Verifica che le colonne esistano
    if 'QUERY' not in df.columns or 'ANSWER' not in df.columns:
        raise ValueError("Il file deve contenere le colonne 'QUERY' e 'ANSWER'")

    # Estrazione dei dati in due liste
    questions = df['QUERY'].tolist()
    answers = df['ANSWER'].tolist()

    return questions,answers
