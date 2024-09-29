from flask import Flask, render_template, request, jsonify

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

# Initialize the tokenizer and model from the DialoGPT-medium checkpoint
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

# Initialize global chat history to maintain the conversation context
chat_history_ids = None

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    global chat_history_ids  

    msg = request.form.get("msg", None)
    
    
    print(f"Received message: {msg}")

    if msg:
        input = msg.strip()  
        if not input:
            return jsonify({"response": "No message received"})

        return jsonify({"response": get_chat_response(input)})
    else:
        return jsonify({"response": "No message received"})

def get_chat_response(text):
    
    input_text = text.lower()

    # Check for specific keywords related to mental health and return a custom response
    if 'anxious' in input_text:
        return "I'm sorry you're feeling anxious. Taking deep breaths and focusing on calming activities might help. Would you like some tips to relax?"
    elif 'depression' in input_text:
        return "I'm really sorry you're feeling this way. Talking to someone or seeking professional help might make a difference. Would you like to know more about resources?"
    elif 'stressed' in input_text:
        return "Feeling stressed can be tough. Make sure you're getting enough rest and be in a calm environment that helps you. Can I help you find some tips for improving sleep?"
    elif 'lonely' in input_text:
        return "Loneliness can feel overwhelming, but you're not alone. Connecting with others, even online, can help. Would you like some ideas on staying connected?"
    elif 'sad' in input_text:
        return "I'm sorry you're feeling sad. It's okay to feel this way sometimes. Would you like to talk about it, or do you need some advice to lift your mood?"
    else:
        # Default response if no keywords are found
        return generate_model_response(text)

def generate_model_response(text):
    global chat_history_ids  # Maintain the chat history between requests
    
    
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

    
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

   
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=150,             
        min_length=20,              
        temperature=0.6,            
        top_k=50,                   
        repetition_penalty=1.2,     
        pad_token_id=tokenizer.eos_token_id
    )

    
    chat_history_ids = chat_history_ids.clone()

    
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run(debug=True)
