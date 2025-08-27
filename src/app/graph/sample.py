import uuid
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent

load_dotenv()
langfuse_handler = CallbackHandler()


class SampleChatBot:
    def __init__(self):
        self.graph = create_react_agent(model=ChatOpenAI(model="gpt-4.1-mini"), tools=[TavilySearch(max_results=2)])
        self.thread_id = str(uuid.uuid4())

    def get_thread_config(self):
        """스레드 설정 반환"""
        return {"configurable": {"thread_id": self.thread_id}}

    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        result = self.graph.invoke(
            {"messages": [("human", message)]},
            config={
                "configurable": {"thread_id": self.thread_id},
                "callbacks": [langfuse_handler],
                "metadata": {"langfuse_tags": ["evaluation", "rag"], "evaluation_run": True},
            },
        )

        return result["messages"][-1].content


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
