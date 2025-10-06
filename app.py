import os
from flask import Flask, render_template, request, jsonify
import whisper
from transformers import pipeline
from datetime import datetime

app = Flask(__name__)

# ---------------------------
# Setup folders and models
# ---------------------------
os.makedirs("recordings", exist_ok=True)

print("Loading Whisper model (base)...")
whisper_model = whisper.load_model("base")  # or "small" for faster CPU

print("Loading summarizer model (facebook/bart-large-cnn)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_audio", methods=["POST"])
def process_audio():
    try:
        audio = request.files.get("audio_data")
        if audio is None:
            return jsonify({"error": "No audio received"}), 400

        # Get filename from popup
        base_filename = request.form.get("filename")
        if not base_filename or not base_filename.strip():
            base_filename = datetime.now().strftime("note_%H-%M-%S")

        # Create folder for today
        today = datetime.now().strftime("%Y-%m-%d")
        date_folder = os.path.join("recordings", today)
        os.makedirs(date_folder, exist_ok=True)

        # Save audio
        audio_filename = base_filename + ".webm"
        audio_path = os.path.join(date_folder, audio_filename)
        audio.save(audio_path)

        # ---------------------------
        # Transcribe with Whisper
        # ---------------------------
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]

        # ---------------------------
        # Summarize with Hugging Face
        # ---------------------------
        input_length = len(transcript.split())
        max_len = min(50, max(10, input_length))
        min_len = min(10, max_len)
        summary = summarizer(
            transcript, max_length=max_len, min_length=min_len, do_sample=False
        )[0]["summary_text"]

        # ---------------------------
        # Save transcript + summary
        # ---------------------------
        text_filename = base_filename + ".txt"
        text_path = os.path.join(date_folder, text_filename)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("=== Transcript ===\n\n")
            f.write(transcript + "\n\n")
            f.write("=== Summary ===\n\n")
            f.write(summary + "\n")

        return jsonify({
            "transcript": transcript,
            "summary": summary
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
