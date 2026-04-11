import streamlit as st
from langGraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid
from datetime import datetime

########################### Utility functions ###########################
def get_thread_id():
    return str(uuid.uuid4())

def generate_thread_name(messages):
    """Generate a meaningful name from the first message or use timestamp"""
    if messages:
        # Use first message as name (truncate if too long)
        first_message = messages[0].get("content", "New Chat")
        return first_message[:50] + "..." if len(first_message) > 50 else first_message
    # Fallback to timestamp
    return f"Chat - {datetime.now().strftime('%H:%M %p')}"

def reset_chat():
    thread_id = get_thread_id()
    st.session_state.thread_id = thread_id
    add_thread(st.session_state.thread_id)
    st.session_state.messages = []

def add_thread(thread_id, name=None):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)
        if name is None:
            name = f"Chat - {datetime.now().strftime('%H:%M %p')}"
        st.session_state.thread_names[thread_id] = name

def load_conversation(thread_id):
    return chatbot.get_state(config = {"configurable": {"thread_id": thread_id}}).values["messages"]


########################### Session State Management ###########################
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = get_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

if "thread_names" not in st.session_state:  # New: store names mapping
    st.session_state.thread_names = {}

add_thread(st.session_state.thread_id)

############################ Sidebar for conversation history ############################

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Conversation History")

for thread_id in st.session_state.chat_threads[::-1]:
    # Display meaningful name instead of thread_id
    display_name = st.session_state.thread_names.get(thread_id, "Chat")
    
    if st.sidebar.button(display_name, key=thread_id):
        st.session_state.thread_id = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": message.content})

        # Update thread name based on first message
        if temp_messages and thread_id not in st.session_state.thread_names:
            name = temp_messages[0]["content"][:50]
            st.session_state.thread_names[thread_id] = name + "..."

        st.session_state.messages = temp_messages

########################## Display chat messages ##########################
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

############################ Chat Input and Response Handling ############################
user_input = st.chat_input("Enter your message here")

if user_input:
    # Update thread name on first message
    if not st.session_state.messages and st.session_state.thread_id in st.session_state.thread_names:
        truncated = user_input[:50] + "..." if len(user_input) > 50 else user_input
        st.session_state.thread_names[st.session_state.thread_id] = truncated
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
                message_chunk.content for message_chunk, metadata in chatbot.stream(
                    input={'messages': [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages"
                )
            )

    st.session_state.messages.append({"role": "assistant", "content": ai_message})
    st.rerun()