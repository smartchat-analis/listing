from flask import Flask, request, jsonify
from main_chatbot import ChatBot

app = Flask(__name__)

# Import main program chatbot
chatbot = ChatBot("conversation_embeddings.json")

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data:
        return jsonify({"error": "JSON tidak boleh kosong"}), 400

    conv_id = data.get("conv_id")
    message = data.get("message")

    if not conv_id or not message:
        return jsonify({
            "error": "conv_id dan message wajib diisi"
        }), 400

    try:
        response = chatbot.chat(message, conv_id)

        return jsonify({
            "status": "success",
            "conv_id": conv_id,
            "response": response
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)