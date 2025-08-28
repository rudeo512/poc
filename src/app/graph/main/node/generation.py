from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.app.graph.main.schema import TaskResult, Task

load_dotenv()


class Generation:
    def __init__(self):
        prompt = """당신은 Task의 실행 결과를 바탕으로 사용자에게 최종 응답을 생성하는 AI 어시스턴트입니다.

아래에 주어진 Task 실행 결과 데이터를 사용하여, 사용자에게 제공할 친절하고 명확한 최종 응답 메시지를 작성해 주세요.

**[Task 실행 결과]**
* **Task 타입 (`type`):** {type}
* **원본 요청 (`question`):** {question}
* **Task 결과 (`result`):** {result}
* **결과 추론 이유 (`result_reasoning`):** {result_reasoning}

**[응답 생성 가이드라인]**
1.  사용자의 원본 요청(`question`)을 자연스럽게 언급하며 답변을 시작하세요.
2.  Task의 최종 결과(`result`)가 있는 경우에만 명확하게 전달하세요.
3.  결과 추론 이유(`result_reasoning`)가 있는 경우에만 결과가 그렇게 나온 이유를 쉽고 논리적으로 설명하여 답변의 신뢰도를 높여주세요.
4.  전체적으로 딱딱하지 않고, 친절하며, 대화하는 듯한 어조를 유지하세요.
5.  `type`, `question`, `result`, `result_reasoning`과 같은 필드 이름을 응답에 직접 노출하지 마세요.

**[생성할 최종 응답]**"""

        prompt_template = PromptTemplate.from_template(
            prompt,
            partial_variables={
                "result": "",
                "result_reasoning": "",
            },
        )
        parser = StrOutputParser()
        llm = ChatOpenAI(model="gpt-4.1-mini")

        self.chain = prompt_template | llm | parser

    async def generate(self, task_result: TaskResult | Task):
        return await self.chain.ainvoke({**task_result.model_dump()})

    def format_docs(self, docs: List[Document]):
        return "\n\n".join([d.page_content for d in docs])


generation = Generation()


async def generate(task_result: TaskResult | Task):
    return await generation.generate(task_result)
