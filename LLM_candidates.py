import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import lmstudio as lms
from openai import OpenAI
from ollama import chat
from ollama import ChatResponse

import json
from tqdm import tqdm

class candidate_answer_generator():
    def __init__(self, read_path, output_path):
        self.output_path=output_path
        with open(read_path, 'r') as readfile:
            num_readfile_lines = sum(1 for line in readfile)
        try:
            with open(output_path, 'r') as outfile:
                num_outfile_lines = sum(1 for line in outfile)
        except:
            num_outfile_lines=0 
        self.data_list = []
        with open(read_path, 'r') as readfile:
            for i, line in enumerate(tqdm(readfile, total=num_readfile_lines)):
                if (i+1)> num_outfile_lines:
                    data = json.loads(line)
                    self.data_list.append(data)
                    

    def llama_answer(self, MODEL_PATH):
        model, tokenizer = load(MODEL_PATH)

        for data in tqdm(self.data_list):

            question = data["question"]
                    
            query = f"Answer the question \"{question}\" in three sentences."
            messages = [{"role": "user", "content": query}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            response = generate(model, tokenizer, prompt=prompt, verbose=False)
            data["candidates"]["llama31_ans"] = response
            with open(self.output_path, 'a') as output_file:
                json.dump(data, output_file, ensure_ascii=False)
                output_file.write('\n')
            
    def qwen_answer(self, MODEL_PATH):
        model, tokenizer = load(MODEL_PATH)

        for data in tqdm(self.data_list):
            question = data["question"]
        
            query = f"Answer the question \"{question}\" in three sentences."
            messages = [{"role": "user", "content": query}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )

            response = generate(model, tokenizer, prompt=prompt, max_tokens=1000)
            
            data["candidates"] = {"qwen_ans":response}
            with open(self.output_path, 'a') as output_file:
                json.dump(data, output_file, ensure_ascii=False)
                output_file.write('\n')

    def gemma3_answer(self, MODEL_PATH):
        model, tokenizer = load(MODEL_PATH)
        for data in tqdm(self.data_list):

            question = data["question"]
            
            query = f"Answer the question \"{question}\" in three sentences."
            messages = [{"role": "user", "content": query}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            response = generate(model, tokenizer, prompt=prompt, verbose=False)
            # print(response)
            data["candidates"]["gemma3_ans"] = response
            with open(self.output_path, 'a') as output_file:
                json.dump(data, output_file, ensure_ascii=False)
                output_file.write('\n')

                    
    def phi4_answer(self, model):
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        for data in tqdm(self.data_list):
            question = data["question"]
            query = f"\"{question}\" Answer the question in a three-sentence paragraph."

            response = client.responses.create(model=model, input=query)
            answer = response.output_text
            
            # print(answer)
            data["candidates"]["phi4_ans"] = answer
            with open(self.output_path, 'a') as output_file:
                json.dump(data, output_file, ensure_ascii=False)
                output_file.write('\n')
        
    def GPT_oss_asnwer(self, model):    
        for data in tqdm(self.data_list):
            question = data["question"]
            
            query = f"\"{question}\" Answer the question in a three-sentence paragraph."
        
            response: ChatResponse = chat(model=model, messages=[
            {
                'role': 'user',
                'content': query,
            },
            ])
            data["candidates"]["GPToos_ans"] = response['message']['content']
            with open(self.output_path, 'a') as output_file:
                json.dump(data, output_file, ensure_ascii=False)
                output_file.write('\n')        
                
    def mistral_asnwer(self, model):    
        for data in tqdm(self.data_list):
            question = data["question"]
            
            query = f"\"{question}\" Answer the question in a three-sentence paragraph."
        
            response: ChatResponse = chat(model=model, messages=[
            {
                'role': 'user',
                'content': query,
            },
            ])
            data["candidates"]["mistral_ans"] = response['message']['content']
            with open(self.output_path, 'a') as output_file:
                json.dump(data, output_file, ensure_ascii=False)
                output_file.write('\n')
                
    def llama32_asnwer(self, model):    
        for data in tqdm(self.data_list):
            question = data["question"]
            
            query = f"\"{question}\" Answer the question in a three-sentence paragraph."
        
            response: ChatResponse = chat(model=model, messages=[
            {
                'role': 'user',
                'content': query,
            },
            ])
            data["candidates"]["llama32_ans"] = response['message']['content']
            with open(self.output_path, 'a') as output_file:
                json.dump(data, output_file, ensure_ascii=False)
                output_file.write('\n')     
                        
    def granite33_asnwer(self, model):    
        for data in tqdm(self.data_list):
            question = data["question"]
            query = f"\"{question}\" Answer the question in a three-sentence paragraph."
        
            response: ChatResponse = chat(model=model, messages=[
            {
                'role': 'user',
                'content': query,
            },
            ])
            # print(response['message']['content'])
            data["candidates"]["granite33_ans"] = response['message']['content']
            with open(self.output_path, 'a') as output_file:
                json.dump(data, output_file, ensure_ascii=False)
                output_file.write('\n')     
                          
    def GPT_oss_asnwer_lm(self, model):
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        for data in tqdm(self.data_list):
            question = data["question"]
            
            query = f"\"{question}\" Answer the question in a three-sentence paragraph."

            response = client.responses.create(model=model, input=query)
            answer = response.output_text

            data["candidates"]["GPToos_ans"] = answer
            with open(self.output_path, 'a') as output_file:
                json.dump(data, output_file, ensure_ascii=False)
                output_file.write('\n')

def main():
    print("-------Qwen3 generate answers----------")
    GEN = candidate_answer_generator(read_path, output_path)
    GEN.qwen_answer("/lmstudio-community/Qwen3-4B-MLX-4bit")
    
    print("-------Phi-4 generate answers----------")
    GEN = candidate_answer_generator(read_path, output_path)
    GEN.phi4_answer("lmstudio-community/phi-4-GGUF")

    print("-------Gemma 3 generate answers----------")
    GEN = candidate_answer_generator(read_path, output_path)
    GEN.gemma3_answer("/mlx-community/gemma-3-27b-it-4bit")
    
    print("-------GPT_oos generate answers----------")
    GEN = candidate_answer_generator(read_path, output_path)
    GEN.GPT_oss_asnwer("gpt-oss")
    
    print("-------llama3.2 generate answers----------")
    GEN = candidate_answer_generator(read_path, output_path)
    GEN.llama32_asnwer("llama3.2:latest")
    
    print("-------mistral-nemo generate answers----------")
    GEN = candidate_answer_generator(read_path, output_path)
    GEN.mistral_asnwer("mistral-nemo:latest")
    
    print("-------granite33 generate answers----------")
    GEN = candidate_answer_generator(read_path, output_path)
    GEN.granite33_asnwer("granite3.3:8b")
    
if __name__ == "__main__":
    main()
