
from pymongo.mongo_client import MongoClient

class MongoDBInterface:
    username = "admin"
    password = "uRQzyxGz95LxSEYW"
    uri = "mongodb+srv://{username}:{password}@explorer-production.oh5gkiu.mongodb.net/?retryWrites=true&w=majority"
    prod_db_name = "explorer-main"

    client = None
    database = None

    @classmethod
    def connect(cls, username: str = None, password: str = None):
        login_uri = cls.uri(
            password = password if password is not None else cls.password,
            username = username if username is not None else cls.username,
        )
        cls.client = MongoClient(login_uri)
        cls.database = cls.client[cls.prod_db_name]

    @classmethod
    def add_items(cls, items: list[dict], collection: str):
        pass