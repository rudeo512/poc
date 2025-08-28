from typing import TypedDict, Literal

from pydantic import BaseModel
from pydantic import Field


class Entity(BaseModel):
    type: Literal["income", "outcome"] = Field(..., description="거래 종류: 수입(income), output(지출)")
    name: str = Field(..., description="거래 항목")
    amount: int = Field(..., description="거래 금액")


class Task(BaseModel):
    type: str = Field(description="task 타입")
    question: str = Field(description="task 내용")
    type_reasoning: str = Field(description="추론 과정 (2 ~ 3줄)")


class TaskResult(Task):
    result: str = Field(description="task 결과")
    result_reasoning: str = Field(description="추론 과정 (2 ~ 3줄)")


class MainAgentInputState(TypedDict):
    question: str


class MainAgentOutputState(TypedDict):
    generation: str


class MainAgentState(MainAgentInputState, MainAgentOutputState):
    task: Task
    task_result: TaskResult

    agreement: bool
    entity: Entity
