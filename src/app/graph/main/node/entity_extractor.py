from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.app.graph.main.schema import Entity

load_dotenv()


class EntityExtractorResponse(BaseModel):
    type: Literal["income", "outcome"] = Field(description="거래 종류: 수입(income), output(지출)")
    name: str = Field(description="거래 항목")
    amount: int = Field(..., description="거래 금액")
    reasoning: str = Field(..., description="추론 과정 (2 ~ 3줄)")


class EntityExtractor:
    def __init__(self):
        prompt = """당신은 질문에서 거래 유형, 항목, 금액을 추출하는 가계부 AI 어시스턴트입니다. 아래 지침에 따라 사용자의 입력을 분석하고 지정된 형식으로 결과를 반환해 주세요.

#지침
1. 거래 타입: 문맥을 파악하여 '수입' 또는 '지출'로 분류합니다.
 - '입금', '받음', '들어옴', '보너스' 등 돈이 들어온 것을 의미하는 단어가 있으면 '수입'으로 판단합니다.
 - 위와 같은 단어 없이 소비를 의미하는 내용이면 '지출'로 판단합니다.
2.항목: 거래의 구체적인 항목을 추출합니다. (예: 외식, 월급, 택시비)
3.금액: 숫자와 단위(원, 만원 등)를 결합하여 숫자만으로 된 금액으로 변환합니다.
 - 금액은 항상 0보다 큰 양수 값이어야 하며, 마이너스(-) 기호는 무시합니다.
 - '5만원' -> 50000
 - '320만원' -> 3200000
 - '12,000원' -> 12000

#예시
입력: 외식 5만원
{{
 거래 타입: 지출
 항목: 외식
 금액: 50000
}}

입력: 아이스크림 구매 1만원
{{
 거래 타입: 지출
 항목: 아이스크림
 금액: 10000
}}

입력: 월급 320만원 입금
{{
 거래 타입: 수입
 항목: 월급
 금액: 3200000
}}

입력: 어제 택시비로 18000원 썼어
{{
 거래 타입: 지출
 항목: 택시비
 금액: 18000
}}

입력: 엄마한테 용돈 10만원 받음
{{
 거래 타입: 수입
 항목: 용돈
 금액: 100000
}}

입력: 친구 선물 사는데 35,000원 지출
{{
 거래 타입: 지출
 항목: 친구 선물
 금액: 35000
}}

#사용자 입력
{question}"""

        prompt_template = PromptTemplate.from_template(prompt)
        llm = ChatOpenAI(model="gpt-4.1-mini").with_structured_output(EntityExtractorResponse)
        self.chain = prompt_template | llm

    async def extract(self, question: str) -> EntityExtractorResponse:
        return await self.chain.ainvoke({"question": question})


entity_extractor = EntityExtractor()


async def entity_extract(question: str) -> Entity:
    response = await entity_extractor.extract(question)
    return Entity(**response.model_dump())


"""
import asyncio


async def test(question: str):
    print(await entity_extract(question))


asyncio.run(test("아메리카노 5000원"))
asyncio.run(test("퇴근 후에 먹는 생맥주 1잔 7000원"))
asyncio.run(test("보너스 3000000"))
asyncio.run(test("급여 4백 5십 만원"))
"""
