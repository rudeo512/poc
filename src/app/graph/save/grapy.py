import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

from src.app.graph.save.node.save_db import db_saver_save
from src.app.graph.save.node.save_embedding import embedding_saver_save
from src.app.graph.save.schema import SaveAgentState, SaveAgentInputState, SaveAgentOutputState
from src.app.graph.schema import AgentContext


def save_db(state: SaveAgentInputState, runtime: Runtime[AgentContext]) -> SaveAgentState:
    db_saver_save(type=state["type"], name=state["name"], amount=state["amount"], user_id=runtime.context.user_id)

    return {"save_db_result": True}


def save_embedding(state: SaveAgentInputState, runtime: Runtime[AgentContext]) -> SaveAgentState:
    embedding_saver_save(
        type=state["type"], name=state["name"], amount=state["amount"], user_id=runtime.context.user_id
    )

    return {"save_embedding_result": True}


def save_check(state: SaveAgentState) -> SaveAgentOutputState:
    return {"save_result": state["save_db_result"] and state["save_embedding_result"]}


# 워크플로우 그래프 초기화
workflow = StateGraph(
    SaveAgentState, input_schema=SaveAgentInputState, output_schema=SaveAgentOutputState, context_schema=AgentContext
)

# 노드 정의
workflow.add_node("save_db", save_db)  # rdb 저장
workflow.add_node("save_embedding", save_embedding)  # embedding 저장
workflow.add_node("save_check", save_check)  # embedding 저장

# 경로 정의
workflow.add_edge(START, "save_db")
workflow.add_edge(START, "save_embedding")
workflow.add_edge(["save_db", "save_embedding"], "save_check")
workflow.add_edge("save_check", END)

# 그래프 컴파일
save_db_graph = workflow.compile(checkpointer=InMemorySaver())

# 그래프 시각화
graph_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(graph_dir, exist_ok=True)
filename = os.path.join(graph_dir, "graph.png")

with open(filename, "wb") as f:
    f.write(save_db_graph.get_graph().draw_mermaid_png())
