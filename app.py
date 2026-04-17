from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get("text")

    result = summarizer(text, max_length=100, min_length=30, do_sample=False)

    return jsonify({
        "summary": result[0]['summary_text']
    })

if __name__ == '__main__':
    app.run(debug=True)