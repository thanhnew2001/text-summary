
from huggingface_hub import snapshot_download
token = "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"

snapshot_download(repo_id="Eugenememe/mix-en-vi-4m", token = token)

from hf_hub_ctranslate2 import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub
from transformers import AutoTokenizer

import shutil
import os

# Define the repository ID on Hugging Face Hub
repo_id = "Eugenememe/mix-en-vi-4m"

# Use snapshot_download to download the repository to the current folder
local_path = snapshot_download(repo_id=repo_id, token=token)

print(f"Repository downloaded to: {local_path}")

# Destination path (current directory)
destination = os.getcwd() + "/mix-en-vi-4m"

# Check if the directory exists, and create it if it does not
if not os.path.exists(destination):
    os.makedirs(destination)
    print(f"Created directory: {destination}")
else:
    print(f"Directory already exists: {destination}")

# Move files from the downloaded directory to the destination
for filename in os.listdir(local_path):
    shutil.move(os.path.join(local_path, filename), os.path.join(destination, filename))

print("Files moved to the current working directory.")



model_name = "mix-en-vi-4m"
# use either TranslatorCT2fromHfHub or GeneratorCT2fromHfHub here, depending on model.
model = TranslatorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name, 
        device="cuda",
        compute_type="int8_float16",
        tokenizer=AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
)
outputs = model.generate(
    text=["How do you call a fast Flan-ingo?", "User: How are you doing?"],
)
print(outputs)
