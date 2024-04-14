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
    return translated_text


perform_translation("Hello how are you", "en-vi")


from transformers import AutoModelWithLMHead, AutoTokenizer

summary_model = "mrm8488/t5-base-finetuned-summarize-news"
tokenizer = AutoTokenizer.from_pretrained(summary_model)
model = AutoModelWithLMHead.from_pretrained(summar_model)

def summarize(text, max_length=150):
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds[0]


