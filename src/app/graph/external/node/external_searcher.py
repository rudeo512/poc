from typing import List

from dotenv import load_dotenv
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document

load_dotenv()


class ExternalSearcher:
    def __init__(self, n: int = 5):
        self.retriever = TavilySearchAPIRetriever(k=n)

    def search(self, question: str) -> str:
        return self.format_docs(self.retriever.invoke(question))

    def format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join([d.page_content for d in docs])


external_searcher = ExternalSearcher()


def external_search(question: str) -> str:
    return external_searcher.search(question)


"""
print(external_search(question="안녕하세요"))
print(external_search(question="오늘날씨 어때요?"))
"""
