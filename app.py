from flask import Flask, render_template, request, jsonify
import torch
import cv2
import numpy as np
import sys
from pathlib import Path

app = Flask(__name__)

# yolov5 로컬 경로를 sys.path에 추가
sys.path.append(str(Path(__file__).resolve().parent / 'yolov5'))

# yolov5 내부 모듈 import
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

# 디바이스 설정
device = 'cpu'

# 한글 수어 모델
korean_model = torch.jit.load('model/korean.pt', map_location=device)
korean_model.eval()
korean_model.names = [
    'giyeok', 'nieun', 'digeut', 'rieul', 'mieum', 'bieup', 'siot', 'ieung',
    'jieut', 'chieut', 'kieuk', 'tieut', 'pieup', 'hieut',
    'a', 'ya', 'eo', 'yeo', 'o', 'yo', 'u', 'yu', 'eu', 'i',
    'ae', 'yae', 'e', 'ye', 'oe', 'wi', 'ui'
]

# 숫자 수어 모델
number_model = torch.jit.load('model/number.pt', map_location=device)
number_model.eval()
number_model.names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# 📌 공통 전처리 함수 (640x640 고정)
def preprocess_image(file):
    img0 = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img0 is None:
        raise ValueError("이미지를 불러오지 못했습니다.")

    img = cv2.resize(img0, (640, 640))  # ✅ 해상도 고정
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB, HWC → CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

# 홈
@app.route('/')
def home():
    return render_template('index.html')

# 퀴즈 페이지
@app.route('/quiz/korean')
def quiz_korean():
    return render_template('quiz_korean.html')

@app.route('/quiz/number')
def quiz_number():
    return render_template('quiz_number.html')

# 한글 퀴즈 예측
@app.route('/predict', methods=['POST'])
def predict_korean():
    file = request.files['image']
    target = request.form.get('target')

    label_map = {
        'giyeok': 'ㄱ', 'nieun': 'ㄴ', 'digeut': 'ㄷ', 'rieul': 'ㄹ',
        'mieum': 'ㅁ', 'bieup': 'ㅂ', 'siot': 'ㅅ', 'ieung': 'ㅇ',
        'jieut': 'ㅈ', 'chieut': 'ㅊ', 'kieuk': 'ㅋ', 'tieut': 'ㅌ',
        'pieup': 'ㅍ', 'hieut': 'ㅎ', 'a': 'ㅏ', 'ya': 'ㅑ',
        'eo': 'ㅓ', 'yeo': 'ㅕ', 'o': 'ㅗ', 'yo': 'ㅛ',
        'u': 'ㅜ', 'yu': 'ㅠ', 'eu': 'ㅡ', 'i': 'ㅣ',
        'ae': 'ㅐ', 'yae': 'ㅒ', 'e': 'ㅔ', 'ye': 'ㅖ',
        'oe': 'ㅚ', 'wi': 'ㅟ', 'ui': 'ㅢ',
    }

    try:
        img = preprocess_image(file)
        pred = korean_model(img)
        pred = non_max_suppression(pred, conf_thres=0.4)[0]

        if pred is not None and len(pred):
            pred = pred.cpu()
            box = max(pred, key=lambda x: x[4])
            cls_id = int(box[5].item())
            conf = float(box[4].item())
            eng_label = korean_model.names[cls_id]
            label = label_map.get(eng_label, 'unknown')
            
        else:
            label = 'none'
            conf = 0.0
            is_correct = False
            
        is_correct = (label == target)
        
        print(f"[KOREAN] target: {target} | 예측 클래스: {label} | confidence: {conf:.3f} | 매칭 결과: {is_correct}")
        
        return jsonify({
            'prediction': label,
            'confidence': round(conf, 3),
            'is_correct': is_correct,
            'target': target
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 숫자 퀴즈 예측
@app.route('/predict_number', methods=['POST'])
def predict_number():
    file = request.files['image']
    target = request.form.get('target')

    try:
        img = preprocess_image(file)
        pred = number_model(img)
        pred = non_max_suppression(pred, conf_thres=0.4)[0]

        if pred is not None and len(pred):
            pred = pred.cpu()
            box = max(pred, key=lambda x: x[4])
            cls_id = int(box[5].item())
            conf = float(box[4].item())
            label = number_model.names[cls_id]

        else:
            label = 'none'
            conf = 0.0
            is_correct = False
            
        is_correct = (label == target)
        
        print(f"[NUMBER] target: {target} | 예측 클래스: {label} | confidence: {conf:.3f} | 매칭: {is_correct}")

        return jsonify({
            'prediction': label,
            'confidence': round(conf, 3),
            'is_correct': is_correct,
            'target': target
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
    
    
