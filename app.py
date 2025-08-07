from flask import Flask
from routes.chat_routes import chat_bp

app = Flask(__name__)
app.register_blueprint(chat_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True)
