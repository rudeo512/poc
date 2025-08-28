from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.app.graph.main.schema import Task

load_dotenv()


class TaskRouterResponse(BaseModel):
    type: Literal["transaction_save_task", "transaction_search_task", "external_search_task", "other_task"] = Field(
        ..., description="Select one of the tasks, based on the user's question."
    )
    reasoning: str = Field(..., description="추론 과정 (2 ~ 3줄)")


class TaskRouter:
    def __init__(self):
        prompt = """You are an AI assistant specializing in routing user questions to the appropriate Tasks.
Use the following guidelines to select the most suitable task(s):

## Task
- **transaction_save_task**: 가계부에 거래내역 등록 요청 질문
- **transaction_search_task**: 가계부에 등록된 거래내역 조회 질문
- **external_search_task**: 외부 검색이 필요한 조회 질문
- **other_task**: 외부 검색이 필요하지 않는 일상적인 질문 포함

Always choose all of the appropriate tasks based on the user's question.

Example:
오늘 날씨 어때?
task -> other

외식 지출 비용 얼마야?
task -> transaction_search

점심 지출 10000원
task -> transaction_save

[question]
{question}
"""

        prompt_template = PromptTemplate.from_template(prompt)
        llm = ChatOpenAI(model="gpt-4.1-mini").with_structured_output(TaskRouterResponse)
        self.chain = prompt_template | llm

    async def ainvoke(self, question: str) -> TaskRouterResponse:
        return await self.chain.ainvoke({"question": question})


task_router = TaskRouter()


async def task_route(question: str) -> Task:
    response = await task_router.ainvoke(question)
    return Task(type=response.type, question=question, type_reasoning=response.reasoning)


"""
import asyncio


async def test(question: str):
    print(await task_route(question))


asyncio.run(test("아메리카노 5000원"))
asyncio.run(test("안녕하세요"))
"""
