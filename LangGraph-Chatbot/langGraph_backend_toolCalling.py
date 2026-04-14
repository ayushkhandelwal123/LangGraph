from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import sqlite3
import requests

# load environment variables from .env file
load_dotenv()

# Define the Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    # repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=2048,
)

# Define the model
model = ChatHuggingFace(llm=llm)

# Define Tools
search_tool = DuckDuckGoSearchRun()

# define a calculator tool using langchain's tool decorator
@tool
def calculator(first_number: float, second_number: float, operation: str) -> dict:
    """
    A simple calculator tool that performs basic arithmetic operations.
    """
    if operation == "add":
        result = first_number + second_number
    elif operation == "subtract":
        result = first_number - second_number
    elif operation == "multiply":
        result = first_number * second_number
    elif operation == "divide":
        if second_number == 0:
            return {"error": "Cannot divide by zero"}
        result = first_number / second_number
    else:
        return {"error": "Invalid operation. Supported operations are add, subtract, multiply, divide."}
    
    return {"first_number": first_number, "second_number": second_number, "operation": operation, "result": result}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=XQ62AN1GQ9CVB3T6"
    response = requests.get(url)
    return response.json()

tools = [search_tool, calculator, get_stock_price]
llm_with_tools = model.bind_tools(tools)

# Define the chatbot state
class ChatbotState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

# Define the chat function that will be used in the graph
def chat(state: ChatbotState):

    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# create a connection to the SQLite database
conn = sqlite3.connect("chatbot_state.db",check_same_thread=False)
# checkpointer
checkpointer = SqliteSaver(conn=conn)

# Define the graph
graph = StateGraph(ChatbotState)

graph.add_node("chat", chat)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
