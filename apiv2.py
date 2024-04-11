from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from markupsafe import escape
from hf_hub_ctranslate2 import GeneratorCT2fromHfHub

app = Flask(__name__)

# Load the model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("minhtoan/t5-small-vietnamese-news")  
# model = AutoModelForSeq2SeqLM.from_pretrained("minhtoan/t5-small-vietnamese-news")
# model.eval()  # Set the model to evaluation mode
# if torch.cuda.is_available():
#     print("cuda is now used")
#     model.cuda()

# from hf_hub_ctranslate2 import GeneratorCT2fromHfHub
# model = GeneratorCT2fromHfHub(
#         # load in int8 on CUDA
#         model_name_or_path=model_name,
#         device="cuda",
#         compute_type="int8_float16",
#         # tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
# )

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("minhtoan/t5-small-vietnamese-news")  
# model_name = AutoModelForSeq2SeqLM.from_pretrained("minhtoan/t5-small-vietnamese-news")

translator = ctranslate2.Translator("t5-small-ct2")
tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")

def t5_summary(input_text):
        translator = ctranslate2.Translator("t5-ct2a")
        tokenizer = transformers.AutoTokenizer.from_pretrained("minhtoan/t5-small-vietnamese-news")
        input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
        
        results = translator.translate_batch([input_tokens])
        
        output_tokens = results[0].hypotheses[0]
        output_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
        
        return output_text


@app.route('/summarize', methods=['GET'])
def summarize_get():
    # Get and sanitize parameters from the request
    text = escape(request.args.get('text', ''))
    max_length = request.args.get('max_length', '150')
    
    # Validate max_length is an integer
    try:
        max_length = int(max_length)
    except ValueError:
        return jsonify({'error': 'max_length must be an integer.'}), 400

    outputs = t5_summary(text)
        
    # Return the summary in JSON format
    return jsonify({'summary': outputs[0]})

@app.route('/summarize', methods=['POST'])
def summarize_post():
    # Parse JSON input
    data = request.get_json(force=True)
    text = escape(data.get('text', ''))
    max_length = data.get('max_length', 150)
    
    # Validate max_length is an integer
    try:
        max_length = int(max_length)
    except ValueError:
        return jsonify({'error': 'max_length must be an integer.'}), 400
    
    outputs = t5_summary(text)
        
    # Return the summary in JSON format
    return jsonify({'summary': outputs[0]})


# not use
def summarize(text, max_length):
    # Ensure text is provided
    if not text:
        return jsonify({'error': 'Text parameter is required.'}), 400

    # Summarize
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    if torch.cuda.is_available():
        tokenized_text = tokenized_text.cuda()
    summary_ids = model.generate(tokenized_text, max_length=max_length)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Return the summary in JSON format
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
