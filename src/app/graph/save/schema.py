from typing import TypedDict, Literal


class SaveAgentInputState(TypedDict):
    type: Literal["income", "outcome"]
    name: str
    amount: int


class SaveAgentOutputState(TypedDict):
    save_result: bool


class SaveAgentState(SaveAgentInputState, SaveAgentOutputState):
    save_db_result: bool
    save_embedding_result: bool
