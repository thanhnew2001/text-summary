import nltk
import re
import os

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


# Load the summarization model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(
    "mrm8488/t5-base-finetuned-summarize-news", legacy=False
)
model_summary = AutoModelForSeq2SeqLM.from_pretrained(
    "mrm8488/t5-base-finetuned-summarize-news"
)

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


def is_references_section(text):
    # Define patterns for detecting citations and reference section titles
    section_titles_pattern = r"(References|Bibliography|Citations|Works Cited)\s*[\n\r]"
    apa_pattern = r"\(.*?\d{4}.*?\)"  # Example: (Author, 2020)
    mla_pattern = r"\w+ \d{4}"  # Example: Author 2020
    harvard_pattern = r"\w+,\s*\w+\.\s*\(\d{4}\)"  # Example: Author, A. (2020)
    chicago_pattern = (
        r"\w+\.\s*\"[^\"]+\"\s*\w+\s*\d{4}"  # Example: Author. "Title" Journal 2020
    )
    simple_citation_pattern = (
        r"([\[\(\{<]\d+[\]\)\}>])\s*(https?://[^\s]+)"  # URL citations with numbers
    )

    # Combine all patterns
    combined_patterns = "|".join(
        [
            section_titles_pattern,
            apa_pattern,
            mla_pattern,
            harvard_pattern,
            chicago_pattern,
            simple_citation_pattern,
        ]
    )

    # Search for all matches in the text
    matches = re.finditer(combined_patterns, text, flags=re.IGNORECASE)

    # Calculate the total length of matched text
    matched_length = sum((match.end() - match.start()) for match in matches)

    # Compare the matched text length to the total text length
    # If matched text is more than 50% of total text, return True
    return matched_length > len(text) * 0.5


def read_file_to_string(filepath):
    """Reads all text from a file and returns it as a single string."""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


def group_sentences(
    sentences: List[str], max_length: int = 384, max_sentences: int = 5
) -> List[str]:
    grouped_sentences = []
    current_group = []
    current_length = 0

    for sentence in sentences:
        if (current_length + len(sentence) > max_length) or (
            len(current_group) >= max_sentences
        ):
            grouped_sentences.append(" ".join(current_group))
            current_group = [sentence]
            current_length = len(sentence)
        else:
            current_group.append(sentence)
            current_length += len(sentence)

    if current_group:
        grouped_sentences.append(" ".join(current_group))

    return grouped_sentences


def translate_text(text, model):
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    translated_sentences = [None] * len(sentences)  # Preallocate list to preserve order

    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(
                lambda s: (
                    model.generate(text=s)
                    if not is_references_section(s)
                    # else s.replace("Trích dẫn:", "Citations:")
                    else ""
                ),
                s,
            ): i
            for i, s in enumerate(sentences)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                translated_sentence = (
                    future.result()
                )  # Obtain the result from the future
            except Exception as exc:
                print(f"Sentence at index {index} generated an exception: {exc}")
                translated_sentences[index] = (
                    f"Error translating sentence: {sentences[index]}"
                )
            else:
                translated_sentences[index] = translated_sentence

    return translated_sentences


def calculate_chunks(sentences):
    try:
        # Tokenize each sentence and store the tokens
        sentence_tokens = [
            tokenizer.encode(sentence, add_special_tokens=True)
            for sentence in sentences
        ]
        total_tokens = sum(len(tokens) for tokens in sentence_tokens)

        # Check if total tokens exceed the maximum allowed for the whole text
        if total_tokens > 2048:
            return []

        # If total tokens are less than 512, just return all sentences joined together
        if total_tokens <= 512:
            return [" ".join(sentences)]

        def split_into_chunks(sentences, sentence_tokens):
            # Recursive function to split sentences into chunks
            if sum(len(tokens) for tokens in sentence_tokens) <= 512:
                return [
                    " ".join(sentences)
                ]  # Return the current set of sentences as a chunk
            else:
                # Divide sentences into halves until chunks are acceptable
                mid_index = len(sentences) // 2
                first_half = sentences[:mid_index]
                second_half = sentences[mid_index:]
                first_tokens = sentence_tokens[:mid_index]
                second_tokens = sentence_tokens[mid_index:]

                return split_into_chunks(first_half, first_tokens) + split_into_chunks(
                    second_half, second_tokens
                )

        return split_into_chunks(sentences, sentence_tokens)
    except Exception as e:
        # In case of any error during the processing, return an empty list
        print(f"An error occurred: {e}")
        return []


def summarize_text(sentences):
    chunks = calculate_chunks(sentences)

    summaries = []
    for chunk in chunks:
        input_ids = tokenizer.encode(
            chunk, return_tensors="pt", add_special_tokens=True
        )
        generated_ids = model_summary.generate(
            input_ids,
            num_beams=4,
            max_length=150,
            repetition_penalty=2.0,
            length_penalty=1,
            early_stopping=True,
        )
        preds = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generated_ids
        ]
        summaries.append(preds[0])

    # Combine and post-process summaries
    summary = " ".join(summaries)
    return summary


def summarize(txt):
    start_time = time()
    txt_en = translate_text(txt, model_vi_en)

    def clean_special_characters(text):
        pattern = r"\[\d+(?:\]\[\d+)*\](?=[.,;:]?(?:\s|$))"
        return re.sub(pattern, "", text)

    sum_txt_en = summarize_text([clean_special_characters(t) for t in txt_en])

    sum_txt_vi = " ".join(translate_text(sum_txt_en, model_en_vi))
    end_time = time()
    total_time = end_time - start_time
    print("Total Summarization time:", total_time, "seconds")

    return sum_txt_en, sum_txt_vi


def main():
    # List of file paths to process
    file_paths = [
        "test/news_ru_t.txt",
        "test/news_ko_t.txt",
        "test/news_uk_t.txt",
        "test/news_us_t.txt",
        "test/news_fr_t.txt",
        "test/news_jp_t.txt",
        "test/news_vi_t.txt",
        "test/novel_t.txt",
    ]

    # Iterate through each file path
    for file_path in file_paths:
        # Read the content from the file
        text_content = read_file_to_string(file_path)

        # Summarize the content (assuming summarize function returns a tuple of English and Vietnamese summaries)
        sum_content_en, sum_content_vi = summarize(text_content)

        # Prepare output file path
        base_name = os.path.basename(file_path)
        name_part = os.path.splitext(base_name)[0]
        output_file_path = os.path.join("test", name_part + "_sum.txt")

        # Write summaries to the output file
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(sum_content_en + "\n")
            file.write(sum_content_vi + "\n")


if __name__ == "__main__":
    main()
