import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# 🎙️ Models
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    chunk_length_s=30,
    return_timestamps=True
)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process_audio", methods=["POST"])
def process_audio():
    try:
        if "audio_data" not in request.files:
            return jsonify({"error": "No audio file received"}), 400

        file = request.files["audio_data"]
        base_name = request.form.get("filename", "New Note").strip() or "New Note"

        date_str = datetime.now().strftime("%Y-%m-%d")
        date_folder = os.path.join(RECORDINGS_DIR, date_str)
        os.makedirs(date_folder, exist_ok=True)

        # Avoid duplicates
        folder_name = base_name
        suffix = 0
        while os.path.exists(os.path.join(date_folder, folder_name)):
            suffix += 1
            folder_name = f"{base_name} {suffix}"

        record_folder = os.path.join(date_folder, folder_name)
        os.makedirs(record_folder, exist_ok=True)

        audio_path = os.path.join(record_folder, folder_name + ".webm")
        file.save(audio_path)

        result = asr(audio_path, return_timestamps=True)
        full_text = " ".join([seg["text"] for seg in result["chunks"]])
        segments = result["chunks"]

        if not full_text.strip():
            return jsonify({"error": "No speech detected"}), 400

        input_len = len(full_text.split())
        max_len = min(max(30, input_len // 2), 150)
        min_len = max(10, input_len // 8)
        raw_summary = summarizer(
            full_text, max_length=max_len, min_length=min_len, do_sample=False
        )[0]["summary_text"]

        summary_sentences = [s.strip() for s in raw_summary.split(".") if s.strip()]
        last_timestamp = segments[-1].get("timestamp", [0, 0])[1] if segments else 0
        total_duration = last_timestamp or 0
        num_points = len(summary_sentences)
        if num_points == 0:
            return jsonify({"error": "No summary sentences generated"}), 400

        timestamps = [(i + 1) * (total_duration / num_points) for i in range(num_points)]

        def format_time(sec):
            m, s = divmod(int(sec), 60)
            return f"{m:02d}:{s:02d}"

        bullet_summary = [
            {"time": format_time(timestamps[i]), "text": summary_sentences[i] + "."}
            for i in range(num_points)
        ]

        # Save transcript + summary
        with open(os.path.join(record_folder, folder_name + "_transcript.txt"), "w", encoding="utf-8") as f:
            f.write(full_text.strip())
        with open(os.path.join(record_folder, folder_name + "_summary.txt"), "w", encoding="utf-8") as f:
            json.dump(bullet_summary, f, ensure_ascii=False, indent=2)

        return jsonify({
            "filename": folder_name,
            "transcript": full_text.strip(),
            "summary": bullet_summary
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/list_recordings")
def list_recordings():
    files = []
    for date_folder in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
        date_path = os.path.join(RECORDINGS_DIR, date_folder)
        if not os.path.isdir(date_path):
            continue
        for subfolder in sorted(os.listdir(date_path)):
            sub_path = os.path.join(date_path, subfolder)
            if os.path.isdir(sub_path):
                files.append({"date": date_folder, "name": subfolder})
    return jsonify(files)


@app.route("/delete_recording", methods=["POST"])
def delete_recording():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    folder_path = os.path.join(RECORDINGS_DIR, filename)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, f))
        os.rmdir(folder_path)

        date_folder = os.path.dirname(folder_path)
        if not os.listdir(date_folder):
            os.rmdir(date_folder)

        return jsonify({"status": "deleted"})
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5050)
