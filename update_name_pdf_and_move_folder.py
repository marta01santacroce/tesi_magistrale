import os
import re
import shutil
import time
from pathlib import Path



def clean_filename_like_gradio(filename):
    name, ext = os.path.splitext(filename)

    # Rimuove caratteri indesiderati
    name = name.replace(",", "")                     # rimuove virgole
    name = re.sub(r"[\[\]\(\)]", "", name)           # rimuove parentesi
    name = re.sub(r"\s+", " ", name).strip()         # spazi multipli -> singolo spazio
    
    return name + ext

def rename_and_move_single_pdf(file_path, destination_folder, max_length=220):

    if not os.path.isfile(file_path):# or not file_path.suffix.lower() != ".pdf":

        print(f"Il file specificato non è valido: {file_path}")
        return

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    folder, original_filename = os.path.split(file_path)
    cleaned_filename = clean_filename_like_gradio(original_filename)

    name, ext = os.path.splitext(cleaned_filename)
    truncated_name = name[:max_length - len(ext)] + ext

    new_filename = truncated_name
    new_path = os.path.join(folder, new_filename)
    destination_path = os.path.join(destination_folder, new_filename)

    # Rinomina se necessario
    if original_filename != new_filename:
        print(f"Rinomino: '{original_filename}' → '{new_filename}'")
        os.rename(file_path, new_path)
    else:
        new_path = file_path  # nessuna rinomina

    # Sposta nella cartella di destinazione
    print(f"Sposto: '{new_filename}' → '{destination_folder}'")
    shutil.move(new_path, destination_path)


def rename_and_move_pdfs(source_folder, destination_folder, max_length=220):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        rename_and_move_single_pdf(file_path=filename, destination_folder=destination_folder)
                                    


