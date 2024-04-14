
from huggingface_hub import snapshot_download
token = "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"

from hf_hub_ctranslate2 import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub
from transformers import AutoTokenizer

import shutil
import os

model_name = "model2"
# use either TranslatorCT2fromHfHub or GeneratorCT2fromHfHub here, depending on model.
model = TranslatorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name, 
        device="cuda",
        compute_type="int8_float16",
        tokenizer=AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
)
outputs = model.generate(
    text=["How do you call a fast Flan-ingo?", "User: How are you doing?"],
)
print(outputs)
