# ✋ 백엔드

Flask 기반의 수어 인식 백엔드 서버입니다.  
YOLOv5로 학습된 TorchScript 모델을 통해 업로드된 이미지를 실시간으로 추론하고, 결과를 JSON으로 반환합니다.  
한글 자모 및 숫자에 대한 수어 인식 퀴즈 기능을 제공합니다.


<br>


## 📂 프로젝트 구조

```bash
backend/
├── app.py          # 메인 서버 실행 파일
├── model/          # 학습된 TorchScript 모델 (korean.pt, number.pt)
└── yolov5/         # YOLOv5 추론용 코드 (커스텀 추론 모듈)
```


<br>


## 🌐 라우팅 정보

| 경로 | 설명 |
|------|------|
| `/` | 홈 화면 |
| `/quiz/korean` | 한글 수어 퀴즈 화면 |
| `/quiz/number` | 숫자 수어 퀴즈 화면 |


<br>


## 🧠 API 사용법

### 📍 POST `/predict` (한글 수어 퀴즈)

- 요청 방식: `multipart/form-data`
- 요청 바디:
  - `image`: 수어 이미지 파일
  - `target`: 문제로 제시된 글자 (ex. "ㅂ")

#### 응답 예시:
```json
{
  "prediction": "ㅂ",
  "confidence": 0.87,
  "target": "ㅂ",
  "is_correct": true
}
```

### 콘솔 출력 예:
```
[KOREAN] target: ㅂ | 예측 클래스: ㅂ | confidence: 0.870 | 매칭 결과: True
```


<br>


### 📍 POST `/predict_number` (숫자 수어 퀴즈)

- 요청 방식: `multipart/form-data`
- 요청 바디:
  - `image`: 수어 이미지 파일
  - `target`: 문제로 제시된 숫자 (ex. "7")

#### 응답 예시:
```json
{
  "prediction": "9",
  "confidence": 0.732,
  "target": "7",
  "is_correct": false
}
```

### 콘솔 출력 예:
```
[NUMBER] target: 7 | 예측 클래스: 9 | confidence: 0.732 | 매칭: False
```


<br>


## ⚙️ 모델 정보

- 모델 형식: TorchScript (.pt)
- 입력 이미지 전처리: `640x640` 크기로 리사이즈 후 정규화
- 사용 디바이스: CPU
- 클래스 목록:
  - 한글: 자음 + 모음 (총 31개)
  - 숫자: 1~10


<br>


## 🔧 주요 구성요소

| 파일/폴더   | 설명 |
|-------------|------|
| `app.py`    | Flask 서버, 라우팅 및 API 정의 |
| `model/`    | `korean.pt`, `number.pt` 모델 파일 저장 |
| `yolov5/`   | YOLOv5 추론을 위한 최소 모듈 (import용) |


<br>


## 👩‍💻 기여자

숙명여자대학교 인공지능공학부 24학번 김수빈

<br>

## 📌 참고사항

- 예측 결과는 JSON으로 반환되며, 정확도(confidence)와 정답 여부(is_correct)를 포함합니다.
- 웹캠 기반 수어 퀴즈 서비스와 연동되어 사용됩니다.
- 모델 인식이 실패한 경우 `label: none`, `confidence: 0.0`으로 응답됩니다.
