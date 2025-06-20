const video = document.getElementById('video');
const targetEl = document.getElementById('target');
const digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'];

let currentTarget = '';
let isLocked = false;

// 웹캠 스트리밍 시작
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream);

// 무작위 숫자 선택
function getRandomDigit() {
  return digits[Math.floor(Math.random() * digits.length)];
}

// 새 문제 설정
function setNewTarget() {
  currentTarget = getRandomDigit();
  targetEl.innerText = currentTarget;
  isLocked = false;
}

// 결과 처리 함수
function processResult(data) {
  console.log("📦 서버 응답 JSON:", data);
  console.log(`🎯 Target: ${currentTarget}`);
  console.log(`🤖 Prediction: ${data.prediction}`);
  console.log(`📊 Confidence: ${data.confidence}`);
  console.log(`✅ Is Correct?: ${data.is_correct}`);

  if (data.is_correct) {
    video.style.border = '5px solid green';
    isLocked = true;
    setTimeout(() => {
      video.style.border = 'none';
      setNewTarget();
    }, 1000);
  } else {
    video.style.border = '5px solid red';
    setTimeout(() => {
      video.style.border = 'none';
    }, 800);
  }
}

// 답안 제출
function submitAnswer() {
  if (isLocked || video.readyState !== 4) return;

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('image', blob, 'capture.jpg');
    formData.append('target', currentTarget);

    fetch('/predict_number', {
      method: 'POST',
      body: formData
    })
    .then(async res => {
      console.log("📥 Response status:", res.status);
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`❌ 서버 오류 ${res.status}:\n${errorText}`);
      }
      return res.json();
    })
    .then(data => {
      processResult(data);
    })
    .catch(err => {
      console.error("❌ 예측 요청 오류:", err);
    });
  }, 'image/jpeg');
}

// 초기 문제 설정
setNewTarget();

// 2초마다 자동 제출
setInterval(() => {
  submitAnswer();
}, 2000);


