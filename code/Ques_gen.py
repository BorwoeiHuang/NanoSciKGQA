import json
import ollama
from neo4j import GraphDatabase
import re
from mlx_lm import load, generate

def data_create(question:str,answer:str):
    chat = {"messages": [
        {"role": "user", "content": f"{question}"},
        {"role": "assistant", "content": f"{answer}."},
        ]}
    output = chat 
    return output

def extract_devices_mlx(question):
    model, tokenizer = load("/mlx-community/gemma-3-12b-it-qat-4bit")
    user_prompt = f"Given text {question}. Extract the full nanoscience concept named entity."
    system_prompt = "You are extracting nanoscience concept named entity from the text."

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=1_000)
    return response

def subgraph_retriever(question: str, graph) ->str:
    result = ""
    entity = extract_devices_mlx(question)
    try:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_system_name', $query, {limit:25}) YIELD node, score
            CALL {
                WITH node
                MATCH (node:nanotechnology|System)-[r]->(m)
                WHERE NOT (m:Date)
                RETURN
                    CASE
                        WHEN m:Property THEN node.name + ' - ' + type(r) + ' -> ' + m.name + ' ' + m.value + ' ' + m.unit
                        ELSE node.name + ' - ' + type(r) + ' -> ' + m.name
                    END AS output
            }
            RETURN output LIMIT 80
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    except:
        print("org.apache.lucene.queryparser.classic.TokenMgrError")
        result=""
    
    return result    

def graphRAG(question: str, graph):
    model, tokenizer = load("/mlx-community/gemma-3-12b-it-qat-4bit")
    
    retrieved_content = subgraph_retriever(question, graph)
    if retrieved_content!="":
        system_prompt = "You are answering questions about nanoscience and nanotechnology from the given graph data."
        user_prompt = f"Given the knowledge graph data {retrieved_content}. Write a paragraph that answers {question}."
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=1_000)
    else:
        response=""
    return response

def get_2hop_2(tx):
    response = tx.run("""
            MATCH (ro:ResearchObjective)
            LIMIT 10
            WITH ro
            MATCH (ro)-[r1]->(m)-[r2]->(n)
            RETURN
                CASE
                    WHEN m:Instrument THEN ro.name + ' research use the instrument ' + m.name  +  ', which '  + type(r2) + ' ' + n.name
                    ELSE ro.name + ' research ' + type(r1) + ' ' + m.name + ' ' + type(r1) + ' ' + m.name  + ', which '  + type(r2) + ' ' + n.name
                END AS output
                
            RETURN output
        """)
    
    return list(response)

def get_2hop(tx):
    response = tx.run("""
            MATCH (nt:nanotechnology)
            LIMIT 10 
            WITH nt
            MATCH (nt)-[r1]->(m)-[r2]->(n)
            RETURN nt.name + type(r1) + ' ' + m.name  + ', which '  + type(r2) + ' ' + n.name
                
        """)
    
    return list(response)

def triplet2hop(driver) ->str:
    with driver.session() as session:
        # response = session.execute_read(get_2hop_and_url)
        response = session.execute_read(get_2hop)
    
    result=""    
    for triplet in response:
        print(triplet)
        result += triplet['output'] +"\n"
    ResearchObjective = triplet["ResearchObjective"]
    URL = triplet["URL"]
            
    return result, ResearchObjective, URL
    

def Q_gen_mlx(triplets): #Used to generate first phase Q
    model, tokenizer = load("/mlx-community/gemma-3-12b-it-qat-4bit")
    
    system_promt = '''You are an expert at generating natural language question from knowledge graph data. Your task is to convert structured knowledge graph triplets into natural, human-readable questions.
        Guidelines:
        - Generate questions that sound natural and conversational
        - Keep questions clear and unambiguous
        - Generate diverse question types for the same information when possible
        \n\n
    '''
    user_prompt = f'''
    Given the following knowledge graph triplets, generate questions:

    Triplets:
    {triplets}

    For each triplet or related set of triplets, generate a natural language question which conatains details and understandable for people who don't have the triplets. 
    '''
    
    prompt = system_promt + user_prompt
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=1_0000)
    try:
        json_pattern = r'\[.*\]'
        Q = re.search(json_pattern, response, re.DOTALL).group(0)
        Q = json.loads(Q)
    except:
        print("JSON format error")
        Q=[]
    return Q

                    
