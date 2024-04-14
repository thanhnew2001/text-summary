from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Ensure NLTK resources are available
nltk.download("punkt", quiet=True)

tokenizer = T5Tokenizer.from_pretrained(
    "mrm8488/t5-base-finetuned-summarize-news", legacy=False
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "mrm8488/t5-base-finetuned-summarize-news"
)


def summarize_text_mt(src):
    start_time = time.time()
    combined_text = " ".join(src)

    def summarize(text, max_length):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        # Chunking sentences to fit within model's token limits, skipping long sentences
        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=True)
            if is_references_section(sentence):
                continue
            if current_length + len(sentence_tokens) <= 512:
                current_chunk.append(sentence)
                current_length += len(sentence_tokens)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence_tokens)
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Function to generate summaries for each chunk
        def generate_summary(chunk):
            input_ids = tokenizer.encode(
                chunk, return_tensors="pt", add_special_tokens=True
            )
            generated_ids = model.generate(
                input_ids,
                num_beams=4,
                max_length=max_length,
                repetition_penalty=2.5,
                length_penalty=0.9,
                early_stopping=True,
            )
            summary = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            return summary

        # Using ThreadPoolExecutor to process chunks in parallel
        summaries = []
        with ThreadPoolExecutor() as executor:
            future_to_summary = {
                executor.submit(generate_summary, chunk): chunk for chunk in chunks
            }
            for future in as_completed(future_to_summary):
                summaries.append(future.result())

        return " ".join(summaries)

    summary = summarize(combined_text, max_length=180)
    end_time = time.time()
    print("Total Summarization time:", end_time - start_time, "seconds")
    return summary


# Example use
text_list = [
    "As Paris gears up for the 2024 Olympics, the city is not just focusing on sports but is also undergoing a significant cultural renaissance. This transformation is set to enrich the already vibrant cultural landscape of the French capital, making the spring and summer of 2024 a period of intense activity and excitement for both locals and visitors.",
    "The city is preparing to welcome a host of new and re-opened cultural venues, promising to add new dimensions to Paris's artistic and cultural wealth. Among these, several museums and cultural institutions are eagerly anticipated to reopen their doors after undergoing renovations. These venues are expected to offer deeper immersions into the history, art, and culture that Paris is renowned for, featuring new exhibits that promise to delight and educate.",
    "In addition to the renovations, Paris is also witnessing the emergence of new cultural concepts. These include innovative restaurants and leisure spaces designed to offer unique experiences to visitors. Such spaces aim to breathe new life into the Parisian scene, enriching the city's dynamism and cultural diversity. This is part of a broader effort to combine tradition and innovation, ensuring that the city remains at the forefront of cultural developments.",
]

print(summarize_text(text_list))
