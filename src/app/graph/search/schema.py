from typing import TypedDict


class SearchAgentInputState(TypedDict):
    question: str


class SearchAgentOutputState(TypedDict):
    result: str


class SearchAgentState(SearchAgentInputState, SearchAgentOutputState):
    pass
