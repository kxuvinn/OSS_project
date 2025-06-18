const video = document.getElementById('video');
const targetEl = document.getElementById('target');
const letters = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ','ㅏ','ㅑ', 'ㅓ','ㅕ', 'ㅗ', 'ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅚ','ㅟ','ㅢ'];

let currentTarget = '';
let isLocked = false;

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream);

function getRandomLetter() {
  return letters[Math.floor(Math.random() * letters.length)];
}
function setNewTarget() {
  targetEl.innerText = getRandomLetter();
}
function setNewTarget() {
     currentTarget = getRandomLetter();
     targetEl.innerText = currentTarget;
     isLocked = false;
  }

setNewTarget();

function submitAnswer() {
    if (isLocked) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
  
    canvas.toBlob(blob => {
      const formData = new FormData();
      formData.append('image', blob, 'capture.jpg');
      formData.append('target', curretnTarget);
  
      fetch('/predict', {
        method: 'POST',
        body: formData
      })
      .then(res => res.json())
      .then(data => {
        if (data.is_corrext) {
            video.style.border = '5px solid green';
            isLocked = true;

            setTimeout(() => {
                video.style.border = 'none';
                setNewTarget();
            }, 1000);
        } else {
            video.style.border = '5px solid red';
        }
      });
    }, 'image/jpeg');
  }
  
  setInterval(() => {
    submitAnswer();
  }, 2000);