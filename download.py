"""
Remember to log in to Hugging Face
enter: huggingface-cli login
enter: hhuggingface token
enter: y

"""

import subprocess
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    # AutoModelWithLMHead,
)

# List of models to convert
models = ["mix-en-vi-4m", "mix-vi-en-4m"]

# Base command without the model name
base_command = "ct2-transformers-converter --model {} --output_dir {} --force --copy_files generation_config.json tokenizer_config.json vocab.json source.spm .gitattributes target.spm --quantization float16"

for model in models:
    # Replace the model placeholder with the current model name
    command = base_command.format(f"Eugenememe/{model}", f"./models/ct2fast-{model}")

    # Execute the command
    subprocess.run(command, shell=True, check=True)

    print(f"Conversion completed for model: {model}")


# # Load the summarization model and tokenizer
# tokenizer = T5Tokenizer.from_pretrained(
#     "mrm8488/t5-base-finetuned-summarize-news", legacy=False
# )
# model_summary = AutoModelForSeq2SeqLM.from_pretrained(
#     "mrm8488/t5-base-finetuned-summarize-news"
# )

# # Save model and tokenizer
# model_summary.save_pretrained("models/t5-base-summarize-news")
# tokenizer.save_pretrained("models/t5-base-summarize-news")
