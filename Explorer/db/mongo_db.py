
from bson import ObjectId
from pymongo.cursor import Cursor
from pymongo.mongo_client import MongoClient

class MongoDBInterface:
    uri = "mongodb+srv://admin:gJZY5tAn3USh5fuI@explorer-production.oh5gkiu.mongodb.net/?retryWrites=true&w=majority"
    prod_db_name = "explorer-main"

    client = None
    database = None

    @classmethod
    def connect(cls, username: str = None, password: str = None):
        cls.client = MongoClient(cls.uri)
        cls.database = cls.client[cls.prod_db_name]

    @classmethod
    def add_items(cls, items: list[dict], collection: str) -> list[ObjectId]:
        if collection not in cls.database.list_collection_names():
            raise NameError(f"Colelction {collection} does not exist in the database")
        result = cls.database[collection].insert_many(items)
        return result.inserted_ids

    @classmethod
    def get_items(cls, filter: dict, collection: str) -> Cursor:
        if collection not in cls.database.list_collection_names():
            raise NameError(f"Colelction {collection} does not exist in the database")
        cursor = cls.database[collection].find(filter)
        return cursor