import nltk
import re
import os
import torch

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

from flask import Flask, request, jsonify
from markupsafe import escape

app = Flask(__name__)

# Load the summarization model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(
    "mrm8488/t5-base-finetuned-summarize-news", legacy=False
)
model_summary = AutoModelForSeq2SeqLM.from_pretrained(
    "mrm8488/t5-base-finetuned-summarize-news"
).to('cuda')



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

def translate(text, model_name):
    # use either TranslatorCT2fromHfHub or GeneratorCT2fromHfHub here, depending on model.
    model = TranslatorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name,
        device="cuda",
        compute_type="int8_float16",
        tokenizer=AutoTokenizer.from_pretrained(model_name)
    )
    outputs = model.generate(
        text=["How do you call a fast Flan-ingo?", "User: How are you doing?"],
    )
    print(outputs)
    return outputs[]


def summarize_text(text, max_length):
    # Ensure text is provided
    if not text:
        return jsonify({'error': 'Text parameter is required.'}), 400

    # Summarize
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    if torch.cuda.is_available():
        tokenized_text = tokenized_text.cuda()
    summary_ids = model_summary.generate(
            tokenized_text,
            num_beams=4,
            max_length=max_length,
            repetition_penalty=2.0,
            length_penalty=1,
            early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route("/translate", methods=["POST"])
def translate_post():
    # Parse JSON input
    data = request.get_json(force=True)
    text = escape(data.get("text", ""))
    model =  escape(data.get("model", model_vi_en))
    max_length = data.get("max_length", 150)

    # Validate max_length is an integer
    try:
        max_length = int(max_length)
    except ValueError:
        return jsonify({"error": "max_length must be an integer."}), 400

    return translate([text], model)

@app.route("/summarize", methods=["POST"])
def summarize_post_en():
    # Parse JSON input
    data = request.get_json(force=True)
    text = escape(data.get("text", ""))
    max_length = data.get("max_length", 150)

    # Validate max_length is an integer
    try:
        max_length = int(max_length)
    except ValueError:
        return jsonify({"error": "max_length must be an integer."}), 400

    return summarize_text(text, max_length)
    
@app.route("/translate_summarize", methods=["POST"])
def summarize_post_vi():
    # Parse JSON input
    data = request.get_json(force=True)
    text = escape(data.get("text", ""))
    max_length = data.get("max_length", 150)

    # Validate max_length is an integer
    try:
        max_length = int(max_length)
    except ValueError:
        return jsonify({"error": "max_length must be an integer."}), 400

    translated_text = translate(text, model_vi_en)
    summarized_text = summarize_text(translated_text, max_length)
    translated_summarized_text = translate(text, model_en_vi)
    return translated_summarized_text

if __name__ == "__main__":
    app.run(debug=True)
