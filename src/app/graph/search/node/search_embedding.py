from typing import List

from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class EmbeddingSearcher:
    def __init__(self):
        pass

    def search(self, user_id: str, question: str) -> str:
        retriever = Chroma(
            collection_name="transaction_db",
            persist_directory="../../../data/chroma_db",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_metadata={"hnsw:space": "cosine"},
        ).as_retriever(search_type="similarity", search_kwargs={"k": 3, "filter": {"user_id": user_id}})

        return self.format_docs(retriever.invoke(question))

    def format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join([d.page_content for d in docs])


embedding_searcher = EmbeddingSearcher()


def embedding_saver_search(user_id: str, question: str) -> str:
    return embedding_searcher.search(user_id, question)


"""
print(embedding_saver_search("user_1", "치킨"))
"""
