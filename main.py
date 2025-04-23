# Import library yang diperlukan
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import logging

import nltk.data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = "sshleifer/distilbart-cnn-12-6"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    logger.info("Model berhasil dimuat")
except Exception as e:
    logger.error(f"Gagal memuat model: {e}")

def summarize_text(text, max_length=150, min_length=40):
    if max_length <= min_length:
        logger.error("max_length harus lebih besar dari min_length")
        return None
    
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    except Exception as e:
        logger.error(f"Error dalam proses summarization: {e}")
        return None

#buat backendnya
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Tidak ada teks yang diberikan", "status": "error"}), 400

        text = data['text']
        max_length = data.get('max_length', 150)
        min_length = data.get('min_length', 40)

        if max_length <= min_length:
            return jsonify({"error": "max_length harus lebih besar dari min_length", "status": "error"}), 400

        summary = summarize_text(text, max_length, min_length)

        if summary:
            return jsonify({"summary": summary, "status": "success"}), 200
        else:
            return jsonify({"error": "Gagal meringkas teks", "status": "error"}), 500

    except Exception as e:
        logger.error(f"Error pada endpoint summarize: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500
    
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API Summarization is running!", "status": "success"}), 200



# server
if __name__ == "__main__":
    port = 5000
    logger.info(f"Menjalankan API server pada port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)


