const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = process.env.PORT || 3000;

// Serve frontend files
app.use(express.static("public"));

// File uploads
const upload = multer({ dest: "uploads/" });

// Dummy audio processing endpoint
app.post("/process_audio", upload.single("audio_data"), (req, res) => {
  const file = req.file;
  const filename = req.body.filename;

  if (!file) {
    return res.json({ error: "No audio file uploaded" });
  }

  // Here you would send `file.path` to your AI transcription/summarization service
  // For now, return dummy transcript & summary
  const transcript = `Transcript of ${filename}`;
  const summary = `Summary of ${filename}`;

  // Optionally delete the uploaded file after processing
  fs.unlink(file.path, (err) => {
    if (err) console.error(err);
  });

  res.json({ transcript, summary });
});

// Start server
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
