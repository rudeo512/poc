import uuid
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv
from langgraph.types import Command

from src.app.graph.main.grapy import graph

load_dotenv()


class SampleChatBot:
    def __init__(self):
        self.graph = graph
        self.thread_id = str(uuid.uuid4())
        self.config = {
            "configurable": {"thread_id": self.thread_id},
        }
        self.user_id = "user_id"

    def get_thread_config(self):
        return {"configurable": {"thread_id": self.thread_id}}

    async def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        return await self._chat(message)

    async def _chat(self, message: str) -> str:
        current_state = self.graph.get_state(self.config)
        if len(current_state.tasks) > 0:
            result = await self.graph.ainvoke(
                Command(resume=message), config=self.config, context={"user_id": self.user_id}
            )
        else:
            result = await self.graph.ainvoke(
                {"question": message}, config=self.config, context={"user_id": self.user_id}
            )

        if "__interrupt__" in result:
            result = result["__interrupt__"][0].value

        return result["generation"]


# Gradio 인터페이스 생성 함수
def create_gradio_interface(chatbot):
    """Gradio 인터페이스를 생성하는 함수"""

    with gr.Blocks(
        title="Sample AI 어시스턴트 (HITL)",
        theme=gr.themes.Soft(),
        css="""
        .chat-container { max-width: 1000px; margin: 0 auto; }
        .example-btn { margin: 2px; }
        """,
        analytics_enabled=False,
    ) as demo:
        gr.ChatInterface(
            fn=chatbot.chat,
            title="",
            description="",
            type="messages",
            textbox=gr.Textbox(placeholder="질문을 입력하세요", container=False, scale=7),
            analytics_enabled=False,
        )

    return demo


# ChatBot 생성
chatbot = SampleChatBot()

# Gradio 인터페이스 생성 및 실행
demo = create_gradio_interface(chatbot)
demo.launch(debug=True, share=False)
