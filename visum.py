
from huggingface_hub import snapshot_download
token = "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"

snapshot_download(repo_id="Eugenememe/mix-en-vi-4m", token = token)

from hf_hub_ctranslate2 import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub
from transformers import AutoTokenizer


model_name = "/root/.cache/huggingface/hub/models--Eugenememe--mix-en-vi-4m/snapshots/f9735b6c75003a36bbabe9d992b8b05625b6b871"
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
