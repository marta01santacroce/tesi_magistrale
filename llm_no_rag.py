from transformers import pipeline
import torch
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage
from transformers import pipeline
import torch


# Inizializza la memoria della conversazione
message_history = ChatMessageHistory()

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

         # Recupera la cronologia della conversazione
        chat_history = "\n".join([
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in message_history.messages
        ])

        # Costruisce dinamicamente il messaggio con la query e la cronologia
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot and an assistant for question-answering tasks. \
                Your goal is to provide helpful, concise, and informative responses while maintaining context also from previous interactions. \
                You should ensure coherence between responses and adapt your style based on user queries.\
                Do NOT repeat the user's question or include unnecessary explanations, only provide a direct answer.",
            },
            {   "role": "user", 
                "content": f"Previous conversation:\n{chat_history}\n---\nNew question: {query}\
                Please provide only the answer, without repeating the question or adding extra commentary."
            },
        ]

        # Genera il prompt con il messaggio aggiornato
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Genera la risposta
        outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.4, top_k=50, top_p=0.95)
        response = outputs[0]["generated_text"].strip()

        # Stampa solo la risposta del modello
        print(response)

        # Aggiorna la cronologia della conversazione
        message_history.add_user_message(query)
        message_history.add_ai_message(response)