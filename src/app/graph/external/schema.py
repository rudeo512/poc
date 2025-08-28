from typing import TypedDict


class OtherAgentInputState(TypedDict):
    question: str


class OtherAgentOutputState(TypedDict):
    result: str


class OtherAgentState(OtherAgentInputState, OtherAgentOutputState):
    pass
