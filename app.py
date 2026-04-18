from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)


summarizer = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base"
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"summary": "Please enter some text."})

       
        result = summarizer(
            "summarize: " + text,
            max_length=80,
            do_sample=False
        )

        summary = result[0]["generated_text"]

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"summary": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
