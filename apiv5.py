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

def escape_quotes(text):
    # Replace single quotes with escaped single quotes
    text = text.replace(chr(39), chr(92) + chr(39))  # 39 is ' and 92 is backslash
    
    # Replace double quotes with escaped double quotes
    text = text.replace(chr(34), chr(92) + chr(34))  # 34 is " and 92 is backslash
    
    return text

# Example usage
sample_text = "This is a 'test' with \"quotes\" to escape."
escaped_text = escape_quotes(sample_text)
print(escaped_text)


def translate(text, model):
    outputs = model.generate(
        text=[text],
    )

    return outputs[0]

print(translate("Liên kết Khách hàng là một dịch vụ được cung cấp bởi Ngân hàng Thế giới cho phép khách hàng đăng ký và truy cập thông tin về các dự án của họ, hồ sơ vay, tình trạng vay, chi tiết giải ngân, phí vay, các giao dịch mua sắm đấu thầu, hiệp định pháp lý và các tài liệu dự án liên quan. Khách hàng có thể sử dụng Liên kết Khách hàng để theo dõi và quản lý các thông tin liên quan đến các khoản vay và dự án của họ, cũng như cung cấp thông tin cần thiết cho việc giải ngân các khoản tiền của dự án trên mạng. Để đăng ký và sử dụng dịch vụ này, khách hàng có thể liên hệ với nhân viên của Ngân hàng Thế giới hoặc truy cập trang web chính thức của Liên kết Khách hàng để yêu cầu đăng ký", model_vi_en))

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
    print(text)
    model_name =  escape(data.get("model_name", "en_vi"))
    max_length = data.get("max_length", 150)

    # Validate max_length is an integer
    try:
        max_length = int(max_length)
    except ValueError:
        return jsonify({"error": "max_length must be an integer."}), 400

    model = None
    if model_name == 'en_vi':
        model = model_en_vi
    else:
        model = model_vi_en
    return translate(text, model)

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

    return summarize_text(text, max_length)
    
@app.route("/translate_summarize", methods=["POST"])
def translate_summarize():
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
    print(translated_text)
    summarized_text = summarize_text(translated_text, max_length)
    print(summarized_text)
    translated_summarized_text = translate(summarized_text, model_en_vi)
    print(translated_summarized_text)
    return translated_summarized_text

if __name__ == "__main__":
    app.run(debug=True, port=5002)
