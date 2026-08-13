# 🌉 SignBridge

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SignBridge** is a powerful, bidirectional sign language communication research prototype. It acts as a bridge between signers and non-signers by offering two highly-optimized, real-time translation pipelines.

Whether you want to type with your hands using our smart Sign-to-Text keyboard or translate your voice into sign language concepts, SignBridge processes everything locally with privacy in mind.

---

## ✨ Key Features

* **🔤 Sign-to-Text Keyboard**: A real-time, conservative alphabet recognizer using a verified MediaPipe + Random Forest pipeline. Includes a 1-second cooldown, custom motion-detection for dynamic letters (`J`, `Z`), and distinct audio feedback.
* **🧠 Smart Autocomplete**: As you sign, a local quantized DistilGPT2 neural model predicts your next words, augmented by Google Web 1T n-gram statistics.
* **🎙️ Voice & Text to Sign**: Speak into your microphone (transcribed locally via Whisper) or type text to generate an ordered sequence of sign concepts. Supports verified native sign videos and deterministic fingerspelling fallbacks.
* **✨ Neon UI**: Features a beautiful, responsive frontend that overlays "cool" neon, glowing, and pulsating skeletal keypoints directly over your live camera feed.
* **🔒 100% Local & Private**: No cloud dependencies. All models, from computer vision to Whisper speech recognition, run locally on your machine.

---

## 🚀 Getting Started (100% Working Setup Guide)

Follow these steps carefully to get SignBridge running locally on your laptop.

### Prerequisites
1. **Python 3.11+**: Make sure Python is installed and added to your system PATH.
2. **Git**: To clone the repository.
3. **FFmpeg** *(Optional but recommended)*: Required if you want to use the Voice-to-Sign microphone feature with Whisper. Download it and add it to your system PATH.

### 1. Clone the Repository
Open your terminal or PowerShell and run:
```bash
git clone https://github.com/Hadi-Mostafa/SignBridge.git
cd SignBridge
```

*(Note: If the repository is inside a larger folder, navigate to the `sign_language_translation` directory).*

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to avoid dependency conflicts.
```powershell
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate
# Or on Mac/Linux:
# source venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Fetch the AI Models
SignBridge uses multiple local models. We provide a script to download and verify them automatically using SHA-256 hashes.
```powershell
python scripts\fetch_models.py

# Verify that all required models loaded successfully
python scripts\doctor.py --load-models
```

*(Optional)* If you want to use the **Voice-to-Sign** feature, download the local Whisper speech model:
```powershell
python scripts\fetch_speech_model.py
```

### 5. Start the Application!
Run the FastAPI backend server:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

🎉 **You're done!** Open your browser and navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🎮 How to Use

### Sign Keyboard (Alphabet Mode)
1. Select **Alphabet Mode** and click **Start Camera**. (Allow camera permissions).
2. Present one handshape at a time to the camera. You will see the neon glowing keypoints map to your hand!
3. The letter will lock in, a short tone will play, and the system will wait 1 second before accepting the next letter.
4. Lower your hand to re-arm repeated letters (e.g., spelling "LL").
5. Use the arrow keys or click to select autocomplete suggestions.

### Voice/Text to Sign
1. Navigate to the second page of the app.
2. Type a sentence or click **Start Recording** to speak into your microphone.
3. The system will process your input and display an ordered sequence of sign concepts. If a native video exists in our database, it will play; otherwise, a deterministic skeletal fallback guide will be shown.

---

## 🛠️ Configuration

You can customize the application by setting environment variables before starting the server. Some useful ones include:

| Variable | Default | Purpose |
|---|---|---|
| `SIGNBRIDGE_OFFLINE` | `false` | Set to `1` to strictly prevent any runtime downloads. |
| `ALPHABET_MIN_CONFIDENCE` | `0.65` | Adjust how strict the model is before accepting a static letter. |
| `WHISPER_MODEL_SIZE` | `base` | Which local speech-recognition model to use. |

---

## 🧪 Running Tests
To verify the integrity of your installation and run regression tests:
```powershell
python -m unittest -v
```

---

## ⚠️ Disclaimer
**SignBridge is an academic research prototype.** It is **not** a certified continuous ASL translator and must **not** be used as a substitute for a qualified human sign language interpreter in legal, medical, or emergency situations. For an in-depth look at our evaluation metrics, domain shifts, and scientific promotion gates, please read the `DIAGNOSTIC_REPORT.md` and `academic_report.md` included in the repository.

---

<div align="center">
  <b>Developed by <a href="https://github.com/Hadi-Mostafa">@Hadi-Mostafa</a></b>
</div>
