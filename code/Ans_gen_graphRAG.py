from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_neo4j import Neo4jVector
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import ollama


def extract_devices(input_text):
    llm_model = "llama3.1"
    llm = ChatOllama(model=llm_model, temperature=0, format="json")

    class Entities(BaseModel):
        """Identifying information about entities."""

        names: list[str] = Field(
            ...,
            description="The system named entity in the text",
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting system named entity from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )


    entity_chain = llm.with_structured_output(Entities)
    output = entity_chain.invoke(input_text)
    return output

def vector_retriever(question: str, vector_index):
    vectorRetriever = vector_index.as_retriever()
    vector_data = [el.page_content for el in vectorRetriever.invoke(question)]
    return vector_data

def graph_retriever(question: str, graph) ->str:
    result = ""
    entities = extract_devices(question)
    for entity in entities.names:
        print(f"++++++++entity name recognized from question: {entity} ++++++++++++")
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_system_name', $query, {limit:25}) YIELD node, score
            CALL {
                WITH node
                MATCH (node)-[r1]->(f1)
                RETURN node.name + ' - ' + type(r1) + ' -> ' + f1.name AS output
                UNION ALL
                WITH node
                MATCH (node)-[r2]->()-[r3]->(f2)
                RETURN node.name + ' - ' + type(r2) + ' - ' + type(r3) + ' -> ' + f2.name AS output
            }
            RETURN output LIMIT 80
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result    


def full_retriever(question: str, graph, vector_index):
    graph_data = graph_retriever(question, graph)
    vector_data = vector_retriever(question, vector_index)
    final_data = f"""
        Graph data: {graph_data} ,
        vector data: {"#Device ". join(vector_data)}
        """

    return final_data

def create_message(message, role):
    return {
        'role': role,
        'content': message
    }

def ask(message):
    chat_messages.append(
        create_message(message, 'user')
    )
    response = ollama.chat(model='phi4', stream=True,  messages=chat_messages)

    assistant_message = ''
    for chunk in response:
        assistant_message += chunk['message']['content']

    chat_messages.append(create_message(assistant_message, 'assistant'))
    return assistant_message

def graphRAG(question: str, graph, vector_index):
    retrieved_content =full_retriever(question, graph, vector_index)
    global chat_messages
    chat_messages = []
    prompt = f"Given the graph data {retrieved_content}. Answer the question {question} in one paragraph based on the information given."
    answer = ask(prompt)
    return answer

if __name__ == "__main__":
    URI = 'bolt://localhost:7687'
    usr = 'neo4j'
    password = '*******'
    my_driver = GraphDatabase.driver(URI, auth=(usr, password))
    
    electronics_graph = Neo4jGraph(
            url=URI,
            username=usr,
            password=password,
        )

    vector_index = Neo4jVector.from_existing_graph(
        embedding = OllamaEmbeddings(model="mxbai-embed-large"),
        url=URI,
        username=usr,
        password=password,
        index_name="device_index",
        search_type="hybrid",
        node_label="Device",
        text_node_properties=["name"],
        embedding_node_property="embedding"
    )

    question = "What are GaAs pseudomorphic HEMT's functions and features?"
    graphRAG(question, electronics_graph, vector_index)
    
