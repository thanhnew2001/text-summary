import nltk
import re
import os

from time import time
from nltk.tokenize import sent_tokenize
from hf_hub_ctranslate2 import TranslatorCT2fromHfHub
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoModelWithLMHead,
)

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

app = Flask(__name__)

# Load the summarization model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(
    "mrm8488/t5-base-finetuned-summarize-news", legacy=False
)
model_summary = AutoModelForSeq2SeqLM.from_pretrained(
    "mrm8488/t5-base-finetuned-summarize-news"
)

model_en_vi = TranslatorCT2fromHfHub(
    model_name_or_path="models/ct2fast-mix-en-vi-4m",
    device="cuda",
    compute_type="int8_float16",
    tokenizer=AutoTokenizer.from_pretrained("models/ct2fast-mix-en-vi-4m"),
)
model_vi_en = TranslatorCT2fromHfHub(
    model_name_or_path="models/ct2fast-mix-vi-en-4m",
    device="cuda",
    compute_type="int8_float16",
    tokenizer=AutoTokenizer.from_pretrained("models/ct2fast-mix-vi-en-4m"),
)

def translate(text, model):
    # Function to split text into chunks of complete sentences with each chunk having less than 500 characters
    def split_text_into_chunks(text, max_length=500):
        sentences = re.split(r'(?<=[.!?]) +', text)  # Split text into sentences
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # Split the text if it's longer than 500 characters
    if len(text) > 500:
        chunks = split_text_into_chunks(text)
    else:
        chunks = [text]

    # Translate each chunk
    translations = model.generate(text=chunks)

    # Combine the translated chunks back into a single string
    translated_text = " ".join(translations)
    
    print(translated_text)
    return translated_text


def summarize_text(text, max_length):
  input_ids = tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True
        )
  generated_ids = model_summary.generate(
            input_ids,
            num_beams=4,
            max_length=max_length,
            repetition_penalty=2.0,
            length_penalty=1,
            early_stopping=True,
        )
  output = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  return output


def summarize(text, max_length):
  translated_text = translate(text, model_vi_en)
  summarized_text = summarize_text(translated_text, max_length)
  translated_summarized_text = translate(text, model_en_vi)
  return translated_summarized_text


@app.route("/summarize", methods=["POST"])
def summarize_post():
    # Parse JSON input
    data = request.get_json(force=True)
    text = escape(data.get("text", ""))
    max_length = data.get("max_length", 150)

    # Validate max_length is an integer
    try:
        max_length = int(max_length)
    except ValueError:
        return jsonify({"error": "max_length must be an integer."}), 400

    return summarize(text, max_length)

if __name__ == "__main__":
    app.run(debug=True)
