from typing import Literal

from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class EmbeddingSaver:
    def __init__(self):
        self.embedding_db = Chroma(
            collection_name="transaction_db",
            persist_directory="../../../data/chroma_db",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_metadata={"hnsw:space": "cosine"},  # l2, ip, cosine 중에서 선택
        )

    def save(self, type: Literal["income", "outcome"], name: str, amount: int, user_id: str):
        self.embedding_db.add_documents(
            [
                Document(
                    page_content=f"{type} {name} {amount}",
                    metadata={
                        "type": type,
                        "name": name,
                        "amount": amount,
                        "user_id": user_id,
                    },
                )
            ]
        )


embedding_saver = EmbeddingSaver()


def embedding_saver_save(type: Literal["income", "outcome"], name: str, amount: int, user_id: str):
    embedding_saver.save(type, name, amount, user_id)


"""
embedding_saver_save("outcome", "치킨", 24000, "user_id1")
embedding_saver_save("outcome", "피자", 14000, "user_id2")
"""
