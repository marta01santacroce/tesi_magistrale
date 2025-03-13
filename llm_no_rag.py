import DB
import search_v2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# Carica il modello
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    return_full_text=False
)

if __name__ == "__main__":
    while True:
        # Query dinamica dall'utente
        query = input("\nEnter your question or press enter without text to exit: ")

        # Se l'utente non inserisce nulla, esce
        if not query.strip():
            print("No query entered. Exiting...")
            break

        # Costruisce dinamicamente il messaggio con la query
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot and an assistant for question-answering tasks",
            },
            {"role": "user", "content": f"Context: {query}"},  # Qui aggiorniamo la query
        ]

        # Genera il prompt con il messaggio aggiornato
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Genera la risposta
        outputs =pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.4, top_k=50, top_p=0.95)

        # Stampa la risposta del modello
        print(outputs[0]["generated_text"])
