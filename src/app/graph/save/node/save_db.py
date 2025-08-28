from datetime import datetime
from typing import Literal

from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase


class DBSaver:
    def __init__(self):
        # SQLite 데이터베이스 연결
        self.db = SQLDatabase.from_uri("postgresql://admin:admin@localhost:5432/db")

    def save(self, type: Literal["income", "outcome"], name: str, amount: int, user_id: str):
        self.db.run(
            f"INSERT INTO transaction (type, name, amount, user_id, created_time) VALUES ('{type}', '{name}', {amount}, '{user_id}', '{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}')"
        )


db_saver = DBSaver()


def db_saver_save(type: Literal["income", "outcome"], name: str, amount: int, user_id: str):
    db_saver.save(type, name, amount, user_id)


"""
db_saver_save("outcome", "치킨", 24000, "user_id1")
db_saver_save("outcome", "피자", 14000, "user_id2")
"""
