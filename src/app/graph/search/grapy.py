import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

from src.app.graph.schema import AgentContext
from src.app.graph.search.node.search_embedding import embedding_saver_search
from src.app.graph.search.schema import SearchAgentInputState, SearchAgentState, SearchAgentOutputState


def search_embedding(state: SearchAgentInputState, runtime: Runtime[AgentContext]) -> SearchAgentOutputState:
    return {"result": embedding_saver_search(runtime.context.user_id, state["question"])}


# 워크플로우 그래프 초기화
workflow = StateGraph(
    SearchAgentState,
    input_schema=SearchAgentInputState,
    output_schema=SearchAgentOutputState,
    context_schema=AgentContext,
)

# 노드 정의
workflow.add_node("search_embedding", search_embedding)  # embedding 조회

# 경로 정의
workflow.add_edge(START, "search_embedding")
workflow.add_edge("search_embedding", END)

# 그래프 컴파일
search_graph = workflow.compile(checkpointer=InMemorySaver())

# 그래프 시각화
graph_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(graph_dir, exist_ok=True)
filename = os.path.join(graph_dir, "graph.png")

with open(filename, "wb") as f:
    f.write(search_graph.get_graph().draw_mermaid_png())
