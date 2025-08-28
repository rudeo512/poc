from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class FeedBackCheckerResponse(BaseModel):
    result: bool = Field(description="질문에 대한 동의 여부")
    reasoning: str = Field(..., description="추론 과정 (2 ~ 3줄)")


class FeedBackChecker:
    def __init__(self):
        prompt = """주어진 질문에 대한 피드백이 명확히 긍정적이고 동의하는 내용일 경우에만 True를,
그렇지 않고 부정적이거나, 애매하거나, 중립적인 모든 경우에는 False를 반환해줘.

질문: {question}
피드백: {feedback}

출력: true/false"""

        prompt_template = PromptTemplate.from_template(prompt)
        llm = ChatOpenAI(model="gpt-4.1-mini").with_structured_output(FeedBackCheckerResponse)

        self.chain = prompt_template | llm

    async def check_feedBack(self, question: str, feedback: str) -> FeedBackCheckerResponse:
        return await self.chain.ainvoke({"question": question, "feedback": feedback})


feedBack_checker = FeedBackChecker()


async def check_feedBack(question: str, feedback: str) -> bool:
    return (await feedBack_checker.check_feedBack(question, feedback)).result


"""
import asyncio


async def test(question: str, feedback: str):
    print(await check_feedBack(question, feedback))


asyncio.run(test("저장 할까요?", "좋아요"))
asyncio.run(test("저장 할까요?", "노"))
asyncio.run(test("저장 할까요?", "잠시만"))
asyncio.run(test("저장 할까요?", "알았어"))
"""
