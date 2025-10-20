import json
from tqdm import tqdm
import re

import ollama

def create_message(message, role):
    return {
        'role': role,
        'content': message
    }

def chat():
    LLM_model = 'gemma3:27b' 
    ollama_response = ollama.chat(model=LLM_model, stream=True, messages=chat_messages)
    assistant_message = ''
    for chunk in ollama_response:
        assistant_message += chunk['message']['content']
    chat_messages.append(create_message(assistant_message, 'assistant'))
    return assistant_message

def ask(message):
    chat_messages.append(
        create_message(message, 'user')
    )
    answer = chat()
    return answer

def gemma3_answer(title, abstract):
    json_pattern = r'\[.*\]'
    while True:
        try:
            global chat_messages
            chat_messages = []
            
            prompt = f"Given a paper's title and abstract from Nano Letters Journal. \n Title: {title}. Abstract: {abstract} \n List all the main device, technology and instrument mentions in this abstract, and their materials, physical mechanism utilized, functions, and features. Answer in JSON format."
            ans= ask(prompt)
            gemma3_extract = re.search(json_pattern, ans, re.DOTALL).group(0)
            gemma3_extract = json.loads(gemma3_extract)
            break
        except:
            print("JSON format error")
            continue
        
    return gemma3_extract
