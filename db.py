from pymongo import MongoClient

# Local MongoDB running on default port
client = MongoClient("mongodb://localhost:27017/")

# Use or create this database
db = client["chat_db"]

# This is the collection that stores all chats
chat_collection = db["chats"]
