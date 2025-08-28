import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.app.graph.external.grapy import other_graph
from src.app.graph.main.node.check_feedback import check_feedBack
from src.app.graph.main.node.entity_extractor import entity_extract
from src.app.graph.main.node.generation import generate
from src.app.graph.main.node.task_router import task_route
from src.app.graph.main.schema import MainAgentInputState, MainAgentState, MainAgentOutputState, TaskResult
from src.app.graph.save.grapy import save_db_graph
from src.app.graph.schema import AgentContext
from src.app.graph.search.grapy import search_graph


async def task_router(state: MainAgentInputState) -> MainAgentState:
    return {"task": await task_route(state["question"])}


async def task_routr_conditional_edges(state: MainAgentState) -> str:
    return state["task"].type


def transaction_save_conditional_edges(state: MainAgentState) -> str:
    return "agreement" if state["agreement"] else "disagreement"


async def transaction_save_task(state: MainAgentState) -> MainAgentState:
    entity = await entity_extract(state["question"])

    feedback = interrupt({"generation": f"거래내역 저장 검토:\n{entity}\n\n저장 할까요?"})
    agreement = await check_feedBack(state["question"], feedback)

    return {
        "agreement": agreement,
        "entity": entity,
        "task_result": (
            None
            if agreement
            else TaskResult(
                **state["task"].model_dump(),
                result="거래내역 저장없이 종료되었습니다.",
                result_reasoning="저장하지 않겠다는 사용자의 직접적인 언급",
            )
        ),
    }


async def transaction_save(state: MainAgentState, runtime: Runtime[AgentContext]) -> MainAgentState:
    await save_db_graph.ainvoke(
        {
            "type": state["entity"].type,
            "name": state["entity"].name,
            "amount": state["entity"].amount,
        },
        context={"user_id": runtime.context.user_id},
    )

    return {
        "task_result": TaskResult(
            **state["task"].model_dump(),
            result=f"{state["entity"].type} {state["entity"].name} {state["entity"].amount} 거래내역이 저장되었습니다.",
            result_reasoning="성공적으로 저장이 되었습니다.",
        )
    }


async def transaction_search_task(state: MainAgentState, runtime: Runtime[AgentContext]) -> MainAgentState:
    result = await search_graph.ainvoke({"question": state["question"]}, context={"user_id": runtime.context.user_id})

    return {
        "task_result": TaskResult(
            **state["task"].model_dump(),
            result=result["result"],
            result_reasoning="성공적으로 검색이 되었습니다.",
        )
    }


async def external_search_task(state: MainAgentState, runtime: Runtime[AgentContext]) -> MainAgentState:
    result = await other_graph.ainvoke({"question": state["question"]}, context={"user_id": runtime.context.user_id})

    return {
        "task_result": TaskResult(
            **state["task"].model_dump(),
            result=result["result"],
            result_reasoning="성공적으로 검색이 되었습니다.",
        )
    }


async def generation(state: MainAgentState) -> MainAgentOutputState:
    response = await generate(state.get("task_result", state["task"]))
    return {"generation": response}


# 워크플로우 그래프 초기화
workflow = StateGraph(
    MainAgentState, input_schema=MainAgentInputState, output_schema=MainAgentOutputState, context_schema=AgentContext
)

# 노드 정의
workflow.add_node("task_router", task_router)  # task 분류

workflow.add_node("transaction_save_task", transaction_save_task)  # 거래내역 등록 Task
workflow.add_node("transaction_search_task", transaction_search_task)  # 거래내역 등록 Task
workflow.add_node("external_search_task", external_search_task)  # 외부 검색 Task
workflow.add_node("transaction_save", transaction_save)  # 거래내역 등록

workflow.add_node("generation", generation)  # 답변 생성

# 경로 정의
workflow.add_edge(START, "task_router")

workflow.add_conditional_edges(
    "task_router",
    task_routr_conditional_edges,
    {
        "transaction_save_task": "transaction_save_task",
        "transaction_search_task": "transaction_search_task",
        "external_search_task": "external_search_task",
        "other_task": "generation",
    },
)

workflow.add_conditional_edges(
    "transaction_save_task",
    transaction_save_conditional_edges,
    {"agreement": "transaction_save", "disagreement": "generation"},
)

workflow.add_edge("transaction_search_task", "generation")
workflow.add_edge("external_search_task", "generation")
workflow.add_edge("transaction_save", "generation")
workflow.add_edge("generation", END)

# 그래프 컴파일
graph = workflow.compile(checkpointer=InMemorySaver())

# 그래프 시각화
graph_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(graph_dir, exist_ok=True)
filename = os.path.join(graph_dir, "graph.png")

with open(filename, "wb") as f:
    f.write(graph.get_graph(xray=True).draw_mermaid_png())
