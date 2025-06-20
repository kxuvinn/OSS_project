const video = document.getElementById('video');
const targetEl = document.getElementById('target');
const letters = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ','ㅏ','ㅑ', 'ㅓ','ㅕ', 'ㅗ', 'ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅚ','ㅟ','ㅢ'];

let currentTarget = '';
let isLocked = false;

// 웹캠 시작
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream);

// 랜덤 문자 설정
function getRandomLetter() {
  return letters[Math.floor(Math.random() * letters.length)];
}

function setNewTarget() {
  currentTarget = getRandomLetter();
  targetEl.innerText = currentTarget;
  isLocked = false;
}

setNewTarget();

// 답안 제출
function submitAnswer() {
  console.log('📸 submitAnswer() called');

  if (isLocked || video.readyState !== 4) return;

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('image', blob, 'capture.jpg');
    formData.append('target', currentTarget);

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(async res => {
      console.log("📥 Response status:", res.status);
      if (!res.ok) {
        const errorText = await res.text();  // HTML 에러 본문
        throw new Error(`❌ 서버 오류 ${res.status}:\n${errorText}`);
      }
      return res.json();
    })
    .then(data => {
      console.log("📦 Parsed JSON:", data);
      console.log(`🎯 Target (문제): ${currentTarget}`);
      console.log(`🤖 Prediction (모델 응답): ${data.prediction}`);
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
    })
    .catch(err => {
      console.error("❌ 예측 요청 오류:", err);
    });
  }, 'image/jpeg');
}

// 2초마다 자동 제출
setInterval(() => {
  submitAnswer();
}, 2000);


