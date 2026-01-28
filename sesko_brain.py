import requests
import json

# This connects to the Ollama brain running on your computer
OLLAMA_URL = "YOUR_API_KEY_HERE"

def ask_sesko(prompt):
    # 1. We give the brain a "Persona" so it knows it is SESKO
    # You can change the 'content' below to change SESKO's personality!
    payload = {
        "model": "llama3.2", 
        "prompt": f"System: You are SESKO, a cool and helpful AI assistant created by Kaif. Answer briefly.\n\nUser: {prompt}",
        "stream": False 
    }

    try:
        # 2. Send message to your local computer brain
        print("SESKO is thinking...") 
        response = requests.post(OLLAMA_URL, json=payload)
        
        # 3. Read the answer
        if response.status_code == 200:
            data = response.json()
            return data["response"]
        else:
            return "Error: My brain isn't responding!"
            
    except Exception as e:
        return f"Connection Error: {e}"

# --- THE CHAT LOOP ---
if __name__ == "__main__":
    print("--- SESKO AI IS ONLINE (LOCAL MODE) ---")
    
    while True:
        # 1. You type a message
        user_input = input("\nYou: ")
        
        # 2. To stop the chat, type 'exit'
        if user_input.lower() == "exit":
            break
            
        # 3. Get the answer
        reply = ask_sesko(user_input)
        
        # 4. Print SESKO's reply
        print(f"SESKO: {reply}")