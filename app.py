from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/quiz/korean')
def quiz_korean():
    return render_template('quiz_korean.html')

@app.route('/quiz/number')
def quiz_number():
    return render_template('quiz_number.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 모델 없이 임시로 정답 여부 랜덤 반환
    is_correct = random.choice([True, False])
    return jsonify({'prediction': 'dummy', 'is_correct': is_correct})

if __name__ == '__main__':
    app.run(debug=True)
