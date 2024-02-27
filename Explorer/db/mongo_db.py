import typing
import pymongo
from pymongo.cursor import Cursor
from bson.objectid import ObjectId


class MongoDBInterface:
    uri = "mongodb+srv://admin:v0jAHE7tGH4RL0fY@explorer-production.oh5gkiu.mongodb.net/?retryWrites=true&w=majority"
    client = None
    database = None

    @classmethod
    def connect(cls):
        cls.client = pymongo.MongoClient(cls.uri)
        cls.client.admin.command("ping")
        cls.database = cls.client["Explorer-production"]

    @classmethod
    def add_items(cls, items: list[dict], collection: str) -> list[ObjectId]:
        if collection not in cls.database.list_collection_names():
            raise NameError(f"Collection {collection} does not exist in the database")
        result = cls.database[collection].insert_many(items)
        return result.inserted_ids

    @classmethod
    def get_items(cls, filter: dict, collection: str) -> Cursor:
        if collection not in cls.database.list_collection_names():
            raise NameError(f"Collection {collection} does not exist in the database")
        cursor = cls.database[collection].find(filter)
        return cursor

    @classmethod
    def delete_items(cls, ids: list[str | ObjectId], collection: str) -> Cursor:
        if collection not in cls.database.list_collection_names():
            raise NameError(f"Collection {collection} does not exist in the database")
        result = cls.database[collection].delete_many({"_id": {"$in": list(map(lambda x: ObjectId(x), ids))}})
        return result
    
    @classmethod
    def delete_all(cls, collection: str) -> Cursor:
        if collection not in cls.database.list_collection_names():
            raise NameError(f"Collection {collection} does not exist in the database")
        result = cls.database[collection].delete_many({})
        return result
