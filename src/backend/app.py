from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from inference import predict_intro
from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
import torch
from flask_socketio import SocketIO, emit
import threading
from transformers import CLIPProcessor, CLIPVisionModel

from feature_extractor import extract_frames_from_video, get_clip_embeddings_for_frames
from inference import predict_intro

app = Flask(__name__, static_folder='../frontend')
socketio = SocketIO(app)

def update_progress(uid, step, total_steps, message):
    percent_complete = int((step / total_steps) * 100)
    socketio.emit("progress_update", {"uid": uid, "progress": percent_complete, "message": message})

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/detect-intro', methods=['POST'])
def detect_intro():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    file_data = request.files['video']
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    uid = str(uuid.uuid4())
    original_path = os.path.join(temp_dir, f"{uid}.mp4")
    file_data.save(original_path)

    def background_task():
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            total_steps = 3
            step = 0

            # Шаг 1: Извлечение кадров
            update_progress(uid, step, total_steps, "🎞 Извлечение кадров")
            frames, timestamps = extract_frames_from_video(original_path, fps_extraction=1, max_duration_sec=180)
            if not frames or len(frames) < 60:
                socketio.emit("progress_update", {"uid": uid, "progress": 100, "message": "❌ Видео слишком короткое"})
                os.remove(original_path)
                return
            step += 1
            update_progress(uid, step, total_steps, "🧠 Извлечение эмбеддингов")

            # Шаг 2: Получение эмбеддингов
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
            clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            embeddings = get_clip_embeddings_for_frames(frames, clip_processor, clip_model, device)
            if embeddings is None:
                socketio.emit("progress_update", {"uid": uid, "progress": 100, "message": "❌ Ошибка извлечения эмбеддингов"})
                os.remove(original_path)
                return
            step += 1
            update_progress(uid, step, total_steps, "🔍 Предсказание...")

            # Шаг 3: Предсказание
            result = predict_intro(
                video_path=original_path,
                model_path="../../models/modelv3.pt",
                device=device
            )

            os.remove(original_path)

            if result:
                start_sec, end_sec = result
                update_progress(uid, total_steps, total_steps, "✅ Готово!")
                socketio.emit("result", {"intro": [start_sec, end_sec], "uid": uid})
            else:
                update_progress(uid, total_steps, total_steps, "❌ Не найдено")
                socketio.emit("result", {"intro": None, "uid": uid})

        except Exception as e:
            print(f"ERROR {e}")
            socketio.emit("progress_update", {"uid": uid, "progress": 100, "message": "❌ Ошибка"})
            socketio.emit("result", {"error": str(e), "uid": uid})


    thread = threading.Thread(target=background_task)
    thread.start()

    return jsonify({"uid": uid})

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)