import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from src.app.graph.external.node.external_searcher import external_search
from src.app.graph.external.schema import OtherAgentState, OtherAgentInputState, OtherAgentOutputState
from src.app.graph.schema import AgentContext


def search(state: OtherAgentInputState) -> OtherAgentOutputState:
    return {"result": external_search(state["question"])}


# 워크플로우 그래프 초기화
workflow = StateGraph(
    OtherAgentState,
    input_schema=OtherAgentInputState,
    output_schema=OtherAgentOutputState,
    context_schema=AgentContext,
)

# 노드 정의
workflow.add_node("search", search)  # embedding 조회

# 경로 정의
workflow.add_edge(START, "search")
workflow.add_edge("search", END)

# 그래프 컴파일
other_graph = workflow.compile(checkpointer=InMemorySaver())

# 그래프 시각화
graph_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(graph_dir, exist_ok=True)
filename = os.path.join(graph_dir, "graph.png")

with open(filename, "wb") as f:
    f.write(other_graph.get_graph().draw_mermaid_png())
