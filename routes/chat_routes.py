from flask import Blueprint, request, jsonify
from db import chat_collection
from datetime import datetime
from main import get_answer_from_pdf  # 👈 import the function
import uuid
import os

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/')
def success():
    return "Chat backend is running ✅"

@chat_bp.route('/chat', methods=['POST'])
def create_chat():
    chat_id = str(uuid.uuid4())
    chat_doc = {
        "chat_id": chat_id,
        "title": request.json.get("title", f"Chat {datetime.utcnow()}"),
        "created_at": datetime.utcnow(),
        "messages": []
    }
    chat_collection.insert_one(chat_doc)
    return jsonify({"chat_id": chat_id, "message": "Chat created"}), 201

@chat_bp.route('/chats', methods=['GET'])
def list_chats():
    chats = list(chat_collection.find({}, {"_id": 0, "chat_id": 1, "title": 1, "created_at": 1}))
    return jsonify(chats)

@chat_bp.route('/chat/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    chat = chat_collection.find_one({"chat_id": chat_id}, {"_id": 0})
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify(chat)

@chat_bp.route('/chat/<chat_id>/message', methods=['POST'])
def add_message(chat_id):
    message = request.json
    message["timestamp"] = datetime.utcnow()

    result = chat_collection.update_one(
        {"chat_id": chat_id},
        {"$push": {"messages": message}}
    )

    if result.modified_count == 0:
        return jsonify({"error": "Chat not found"}), 404

    return jsonify({"message": "Message added"}), 200

# ✅ New: Ask a question and store both question + answer
@chat_bp.route('/chat/<chat_id>/ask', methods=['POST'])
def ask_question(chat_id):
    data = request.json
    question = data.get("question")
    pdf_path = data.get("pdf_path", "abc.pdf")  # default fallback

    if not question:
        return jsonify({"error": "Missing question"}), 400

    answer = get_answer_from_pdf(pdf_path, question)

    # Add to DB
    message_doc = {
        "role": "user",
        "content": question,
        "timestamp": datetime.utcnow()
    }
    answer_doc = {
        "role": "agent",
        "content": answer,
        "timestamp": datetime.utcnow()
    }

    chat_collection.update_one(
        {"chat_id": chat_id},
        {"$push": {"messages": {"$each": [message_doc, answer_doc]}}}
    )

    return jsonify({"question": question, "answer": answer}), 200
