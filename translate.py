from hf_hub_ctranslate2 import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub
from transformers import AutoTokenizer


def perform_translation(text, model_dir):
    model = TranslatorCT2fromHfHub(
            model_name_or_path=model_dir,
            device="cuda",
            compute_type="int8_float16",
            tokenizer=AutoTokenizer.from_pretrained(model_dir),
        )

    translated_text = model.generate(text=text)
    return translated_text, end_time - start_time


perform_translation("Hello how are you", "en-vi")
