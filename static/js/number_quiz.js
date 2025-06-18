const video = document.getElementById('video');
const targetEl = document.getElementById('target');
const digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'];

let currentTarget = '';
let isLocked = false;

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream);

function getRandomDigit() {
  return digits[Math.floor(Math.random() * digits.length)];
}

function setNewTarget() {
  currentTarget = getRandomDigit();
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
    formData.append('target', currentTarget); 

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      if (data.is_correct) {
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
