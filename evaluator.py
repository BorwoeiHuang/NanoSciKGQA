from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torch
from ollama import chat
from ollama import ChatResponse
from qa_metrics.f1 import f1_score_with_precision_recall
import evaluate

from tqdm import tqdm
import json
import re



def mistral_evaluator(model, question, reference, candidate):

    system_prompt = '''
        You are an expert Nanoscience and Nanotechnology Science professor and grader. Your task is to objectively compare a Candidate Answer to a Reference Answer and assign sub-scores for specific criteria.   
        Instructions
        Domain Focus: Evaluate the technical accuracy, precision, and completeness of the content strictly within the context of nanoscience, quantum mechanics, material properties, synthesis, and characterization techniques (e.g., TEM, STM, XRD, AFM).

        Scoring Scale: Use a scale of 0.0 to 1.0 for each sub-score, where 1.0 is perfect agreement/coverage and 0.0 is no agreement/coverage.

        Strictness: The candidate must use correct nanoscience terminology and principles. Incorrect or misleading technical claims, even if seemingly related, should severely reduce the score for the relevant criteria.

        Output Format: You must provide your final output exclusively in the specified JSON format.

    '''
    user_prompt1=f"""
        Domain: Nanoscience and nanotechnology 
        
        Question:
        {question}

        Reference answer:
        {reference}

        Candidate answer:
        {candidate}
        
        Evaluation Criteria:
        1. Technical Accuracy: Measures the correctness of all scientific concepts, definitions (e.g., quantum confinement, excitons), and equations used. Are there any factual errors or misstatements of nanoscience principles?
        2. Concept Coverage: Measures how well the candidate's answer addresses all core concepts and components present in the Reference Answer. Is the answer complete?
        3. Domain Specificity: Measures the appropriate use of specific nanoscience and nanotechnology (e.g., "colloidal synthesis," "surface-to-volume ratio," "Bragg's law," "plasmon resonance"). Is the language precise?
        4. Clarity & Coherence: Measures the logical flow, structure, and clarity of the explanation. Is the answer easy to follow and professionally written?

    """
    user_prompt2="""  
        Return your evaluation in JSON format:  
        {"technical_accuracy": X,
        "concept_coverage": X,
        "domain_specificity": X,
        "clarity_coherence": X
        }
    """
    user_prompt = user_prompt1+user_prompt2

    for i in range(3):
        response: ChatResponse = chat(model=model, messages=[
                {
                    'role': 'user',
                    'content': system_prompt+user_prompt,
                },
                ])
        answer = response['message']['content']
        try:
            
            if re.search(r'"technical_accuracy":\s*(-?\d+\.?\d*)', answer, re.DOTALL):
                technical_accuracy = re.search(r'"technical_accuracy":\s*(-?\d+\.?\d*)', answer, re.DOTALL).group(1)
                concept_coverage = re.search(r'"concept_coverage":\s*(-?\d+\.?\d*)', answer, re.DOTALL).group(1)
                domain_specificity = re.search(r'"domain_specificity":\s*(-?\d+\.?\d*)', answer, re.DOTALL).group(1)
                clarity_coherence = re.search(r'"clarity_coherence":\s*(-?\d+\.?\d*)', answer, re.DOTALL).group(1) 
            elif re.search(r'"score":\s*(-?\d+\.?\d*)', answer, re.DOTALL):
                scores = re.findall(r'"score":\s*(-?\d+\.?\d*)', answer)
                technical_accuracy = scores[0]
                concept_coverage = scores[1]
                domain_specificity = scores[2]
                clarity_coherence = scores[3]
            elif re.search(r'"candidate":\s*(-?\d+\.?\d*)', answer, re.DOTALL):
                scores = re.findall(r'"candidate":\s*(-?\d+\.?\d*)', answer)
                technical_accuracy = scores[0]
                concept_coverage = scores[1]
                domain_specificity = scores[2]
                clarity_coherence = scores[3]
                
                
            LLM_eval = {"technical_accuracy": float(technical_accuracy),
                        "concept_coverage": float(concept_coverage),
                        "domain_specificity": float(domain_specificity),
                        "clarity_coherence": float(clarity_coherence)}
            
            return LLM_eval
        except:
            print("JSON format error")
            print(response)
            LLM_eval={}
    return LLM_eval
 
class BERT_evaluator:
    def __init__(self, model, tokenizer, candidate, reference):
        cand_tokenized = tokenizer(candidate, padding=True, truncation=True, return_tensors='pt')
        ref_tokenized = tokenizer(reference, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            self.can_encode = model(**cand_tokenized)
            self.ref_encode = model(**ref_tokenized)
    
    def recall_BERT(self ):#Recall BERT
        sum = 0    
        ref_token_length = len(self.ref_encode["last_hidden_state"][0])
        for ref_vector in self.ref_encode["last_hidden_state"][0]:
            inner_product_list = []
            for gen_vector in self.can_encode["last_hidden_state"][0]:
                inner_product= F.cosine_similarity(ref_vector, gen_vector, dim=0).item()
                inner_product_list.append(inner_product)
            most_similar = max(inner_product_list)
            sum+=most_similar
        R_BERT = sum/ref_token_length
        return R_BERT

    
    def precision_BERT(self):#Precision BERT
        sum = 0        
        gen_token_length = len(self.can_encode["last_hidden_state"][0])
            
        for gen_vector in self.can_encode["last_hidden_state"][0]:
            inner_product_list = []
            for ref_vector in self.ref_encode["last_hidden_state"][0]:
                inner_product= F.cosine_similarity(ref_vector, gen_vector, dim=0).item()
                inner_product_list.append(inner_product)
            most_similar = max(inner_product_list)
            sum+=most_similar
        P_BERT = sum/gen_token_length
        return P_BERT

    def f1_BERT(self):   
        inner_product_arrays = []
        for ref_vector in self.ref_encode["last_hidden_state"][0]:
            ref_inner_product_list = []
            for gen_vector in self.can_encode["last_hidden_state"][0]:
                inner_product= F.cosine_similarity(ref_vector, gen_vector, dim=0).item()
                ref_inner_product_list.append(inner_product)
            inner_product_arrays.append(ref_inner_product_list) 
        num_rows = len(inner_product_arrays)
        num_cols = len(inner_product_arrays[0])
        row_maxes = []
        for row in inner_product_arrays:
            row_max = max(row)
            row_maxes.append(row_max)
        R_BERT = sum(row_maxes)/num_rows
        
        column_maxes = []
        for j in range(num_cols):
            column_elements = [inner_product_arrays[i][j] for i in range(num_rows)]
            column_max = max(column_elements)
            column_maxes.append(column_max)
        P_BERT = sum(column_maxes)/num_cols
        F_BERT = 2*(R_BERT * P_BERT)/(R_BERT + P_BERT)
        return R_BERT, P_BERT, F_BERT

class full_eval:
    def __init__(self, Q_candA_list):
        self.Q_candA_list= Q_candA_list
        self.num_quest= len(Q_candA_list)
            
    def f1_full_eval(self, output_folder):
        output_path = f"{output_folder}/f1.json"
        print("----f1 evaluation--")
        print(f"----output file: {output_path}")
        f1_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0, "llama32_ans":0, "mistral_ans":0, "granite33_ans":0}     
        R_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0, "llama32_ans":0, "mistral_ans":0, "granite33_ans":0}  
        P_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0, "llama32_ans":0, "mistral_ans":0, "granite33_ans":0}  
        
        f1_eval=[]
        for Q_candA in tqdm(self.Q_candA_list):

            reference = Q_candA["reference"]
            candidates = Q_candA["candidates"]
            f1_scores={}
            for cand in candidates:
                answer = candidates[cand]
                f1_score = f1_score_with_precision_recall(reference, answer)
                f1_scores[cand] = f1_score
                f1_avg[cand] += f1_score["f1"]
                P_avg[cand] += f1_score["precision"]
                R_avg[cand] += f1_score["recall"]

            f1_eval.append({"id": Q_candA["id"], "question":Q_candA["question"], "type":Q_candA["type"], "f1_scores":f1_scores})
        f1_avg = {key: value / self.num_quest for key, value in f1_avg.items()}
        P_avg = {key: value / self.num_quest for key, value in P_avg.items()}
        R_avg = {key: value / self.num_quest for key, value in R_avg.items()}
        output_dict = {"f1_avg":f1_avg, "R_avg":R_avg, "P_avg":P_avg, "f1_eval":f1_eval}

    def rouge1_full_eval(self, output_folder):
        output_path = f"{output_folder}/rouge1.json"
        print("----rouge1 evaluation--")
        print(f"----output file: {output_path}")
        rouge = evaluate.load('rouge')
    
        rouge1_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0, "llama32_ans":0, "mistral_ans":0, "granite33_ans":0}  
        
        rouge1_eval=[]
        for Q_candA in tqdm(self.Q_candA_list):
            id = Q_candA["id"]
            question = Q_candA["question"]
            reference = Q_candA["reference"]
            candidates = Q_candA["candidates"]
            rouge1_scores={}
            for cand in candidates:
                answer = candidates[cand]
                rouge1 = rouge.compute(predictions=[answer],references=[reference])["rouge1"]
                rouge1_scores[cand] = rouge1
                rouge1_avg[cand] += rouge1
            rouge1_eval.append({"id": id, "question":question, "rouge1_scores":rouge1_scores})
        rouge1_avg = {key: value / self.num_quest for key, value in rouge1_avg.items()}
        output_dict = {"rouge1_avg":rouge1_avg, "rouge1_eval":rouge1_eval}


    def BERT_full_eval(self, output_folder):
        output_path = f"{output_folder}/BertRecall.json"
        print("----BertRecall evaluation--")
        print(f"----output file: {output_path}")
        f1_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0, "llama32_ans":0, "mistral_ans":0, "granite33_ans":0}      
        R_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0, "llama32_ans":0, "mistral_ans":0, "granite33_ans":0}  
        P_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0, "llama32_ans":0, "mistral_ans":0, "granite33_ans":0}  
        
        model_name = "sentence-transformers/all-roberta-large-v1"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)    
        

        BertRecall_eval=[]
        for Q_candA in tqdm(self.Q_candA_list):
            id = Q_candA["id"]
            question = Q_candA["question"]
            reference = Q_candA["reference"]
            candidates = Q_candA["candidates"]
            BertRecall_scores={}
            for cand in candidates:
                answer = candidates[cand]
                evaluator = BERT_evaluator(model, tokenizer, answer, reference)
                recall, precision, f1 =  evaluator.f1_BERT()
                BertRecall_scores[cand] = {"f1":f1, "precision":precision, "recall":recall}
                
                f1_avg[cand] += f1
                P_avg[cand] += precision
                R_avg[cand] += recall
            BertRecall_eval.append({"id": id, "question":question, "question":question, "BertRecall_scores":BertRecall_scores})
        f1_avg = {key: value / self.num_quest for key, value in f1_avg.items()}
        P_avg = {key: value / self.num_quest for key, value in P_avg.items()}
        R_avg = {key: value / self.num_quest for key, value in R_avg.items()}
        output_dict = {"f1_avg":f1_avg, "R_avg":R_avg, "P_avg":P_avg, "BertRecall_eval":BertRecall_eval}

    def mistral_full_eval(self, output_folder):
        output_path = f"{output_folder}/mistral.json"
        print("----LLM mistral evaluation--")
        print(f"----output file: {output_path}")
        try:
            with open(output_path, "r") as f:
                processedData = json.load(f)
            mistral_eval = processedData["mistral_eval"]
        except:
            mistral_eval=[]
        
        
        num_processed = len(mistral_eval)
        for i, Q_candA in tqdm(enumerate(self.Q_candA_list)):
            if (i+1)>num_processed:
                id = Q_candA["id"]
                question = Q_candA["question"]
                reference = Q_candA["reference"]
                candidates = Q_candA["candidates"]
                mistral_scores={}
                for cand in candidates:
                    answer = candidates[cand]
                    scores = mistral_evaluator("mistral-nemo:latest", question, reference, answer)
                    mistral_scores[cand] = scores
                    
                mistral_eval.append({"id": id, "question":question, "question":question, "type": Q_candA["type"], "mistral_scores":mistral_scores})
                output_dict = {"mistral_eval":mistral_eval}
