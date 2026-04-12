from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import sqlite3

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

class ChatbotState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

def chat(state: ChatbotState):

    response = model.invoke(state["messages"])
    return {"messages" : [response]}

# create a connection to the SQLite database
conn = sqlite3.connect("chatbot_state.db",check_same_thread=False)
# checkpointer
checkpointer = SqliteSaver(conn=conn)

# Define the graph
graph = StateGraph(ChatbotState)

graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpointer)

# using stream so that the AI response loads line by line for better ux
# for message_chunk, metadata in chatbot.stream(
#     input = {'messages': [HumanMessage(content="What is the recipe to make a sandwich?")]},
#     config = {"configurable": {"thread_id": "thread_1"}},
#     stream_mode = "messages"
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=" ", flush=True)

# config = {"configurable": {"thread_id": "thread_1"}}
# response = chatbot.invoke(
#     input = {'messages': [HumanMessage(content="What is the recipe to make a sandwich?")]},
#     config = config,
# )

# print(chatbot.get_state(config=config).values["messages"])


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
