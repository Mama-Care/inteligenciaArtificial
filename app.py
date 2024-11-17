from flask import Flask, request, jsonify
from chat import process_question

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def answer_question():
    # Recebe a pergunta do JSON enviado no POST
    data = request.get_json()
    print(f"BACKEND: { data }")
    question = data.get('question', '')
    
    # Aqui, você processa a pergunta
    response = process_question(question)

    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
