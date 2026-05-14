from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from typing import TypedDict, List, Optional, Annotated
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

load_dotenv()

# define the hugging face endpoint
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=2048,
)

# define the model
model = ChatHuggingFace(llm=llm)

class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="What to cover in this section")
    content: Optional[str] = Field(None, description="The main content of the section")
    section_len
