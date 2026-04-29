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

class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="What to cover in this section")

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str

# define the hugging face endpoint
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=2048,
)

# define the model
model = ChatHuggingFace(llm=llm)

def orchestrator(state: State):
    parser = PydanticOutputParser(pydantic_object=Plan)
    plan = model.invoke(
        [
            SystemMessage(
                content=(f"Create a blog plan with 5 to 7 sections based on the given topic.\n{parser.get_format_instructions()}")
            ),
            HumanMessage(content=f"Topic: {state['topic']}")
        ]
    )

    return {"plan": parser.parse(plan.content.strip())}

def fanout(state: State):
    return [Send("worker", {"task": task, "topic": state["topic"], "plan": state["plan"]})
            for task in state["plan"].tasks]

def worker(payload: dict) -> dict:

    # payload contains what we sent
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    blog_title = plan.blog_title

    section_md = model.invoke(
        [
            SystemMessage(content="Write one clean Markdown section."),
            HumanMessage(
                content=(
                    f"Blog: {blog_title}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Brief: {task.brief}\n\n"
                    "Return only the section content in Markdown."
                )
            ),
        ]
    ).content.strip()

    return {"sections": [section_md]}

from pathlib import Path
import re

def reducer(state: State) -> dict:
    
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    # ---- save to file ----
    # Sanitize filename: remove/replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', title.lower().replace(" ", "_"))
    filename = filename + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return {"final": final_md}

g = StateGraph(State)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

out = app.invoke({"topic": "Write a blog on Self Attention", "sections": []})
print(out["final"])