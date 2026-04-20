from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=2048,
)

# Define the model
model = ChatHuggingFace(llm=llm)

class SubState(TypedDict):
    input: str
    translated_text: str

def translate(state: SubState):
    prompt = f"""
You are a helpful assistant that translates English to Hindi.
Translate the following English text to Hindi:
{state['input']}
"""
    
    response = model.invoke(prompt).content.strip()
    return {"translated_text": response}

# Define the subgraph
subgraph = StateGraph(SubState)
subgraph.add_node("translate", translate)
subgraph.add_edge(START, "translate")
subgraph.add_edge("translate", END)

subgraph_workflow = subgraph.compile()

class ParentState(TypedDict):
    question: str
    english_response: str
    hindi_response: str

def generate_response(state: ParentState):
    prompt = f"""
You are a helpful assistant that answers questions in English.
Answer clearly: {state['question']}
"""
    response = model.invoke(prompt).content.strip()
    return {"english_response": response}

def translate_answer(state: ParentState):
    # Call the subgraph to translate the English response to Hindi
    result = subgraph_workflow.invoke({"input": state['english_response']})
    return {"hindi_response": result['translated_text']}

# Define the parent graph
parent_graph = StateGraph(ParentState)
parent_graph.add_node("generate_response", generate_response)
parent_graph.add_node("translate_answer", translate_answer)
parent_graph.add_edge(START, "generate_response")
parent_graph.add_edge("generate_response", "translate_answer")
parent_graph.add_edge("translate_answer", END)

parent_workflow = parent_graph.compile()

final_result = parent_workflow.invoke({"question": "What are transformers in NLP?"})
print("English Response:", final_result['english_response'])
print("Hindi Response:", final_result['hindi_response'])
