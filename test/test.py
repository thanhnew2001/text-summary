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
    # AutoModelWithLMHead,
)

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


# Load the summarization model and tokenizer

tokenizer = T5Tokenizer.from_pretrained(
    "mrm8488/t5-base-finetuned-summarize-news", legacy=False
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "mrm8488/t5-base-finetuned-summarize-news"
)


# summary_model_name = "mrm8488/t5-base-finetuned-summarize-news"
# tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
# model = AutoModelWithLMHead.from_pretrained(summary_model_name)


model_en_vi = TranslatorCT2fromHfHub(
    model_name_or_path="models/ct2fast-mix-en-vi-4m",
    device="cuda",
    compute_type="int8_float16",
    tokenizer=AutoTokenizer.from_pretrained("models/ct2fast-mix-en-vi-4m"),
)

model_vi_en = TranslatorCT2fromHfHub(
    model_name_or_path="models/ct2fast-mix-vi-en-1m",
    device="cuda",
    compute_type="int8_float16",
    tokenizer=AutoTokenizer.from_pretrained("models/ct2fast-mix-vi-en-1m"),
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


def process_text(text):
    """Processes the given text into sentences and prints statistics about them."""
    # Tokenize the text into sentences using NLTK
    sentences = sent_tokenize(text)

    # Removing empty sentences just in case
    sentences = [s.strip() for s in sentences if s.strip()]

    # Print snippets of each sentence (first 20 characters)
    for sentence in sentences:
        print(f"[{is_references_section(sentence)}]\n{sentence}")

    # Statistics
    num_sentences = len(sentences)
    words = [word for sentence in sentences for word in sentence.split()]
    num_words = len(words)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    max_length = max(sentence_lengths)
    min_length = min(sentence_lengths)
    sentence_char_lengths = [len(sentence) for sentence in sentences]
    max_length_char = max(sentence_char_lengths)
    min_length_char = min(sentence_char_lengths)
    avg_length = sum(sentence_lengths) / num_sentences if num_sentences > 0 else 0

    # Printing results
    print(f"Total number of sentences: {num_sentences}")
    print(f"Total number of words: {num_words}")
    print(f"Maximum sentence length: {max_length} words")
    print(f"Minimum sentence length: {min_length} words")
    print(f"Average sentence length: {avg_length:.2f} words")
    print(f"Maximum sentence length by characters: {max_length_char} characters")
    print(f"Minimum sentence length by characters: {min_length_char} characters")

    # if num_sentences > 0:
    #     shortest_sentence = min(sentences, key=len)
    #     longest_sentence = max(sentences, key=len)


# unused for now
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


def translate_text_to_english(text):
    start_time = time()
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    translated_sentences = [None] * len(sentences)  # Preallocate list to preserve order

    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(
                lambda s: (
                    model_vi_en.generate(text=s)
                    if not is_references_section(s)
                    else s.replace("Trích dẫn:", "Citations:")
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

    end_time = time()
    total_time = end_time - start_time
    print("Total translation time:", total_time, "seconds")

    return translated_sentences


def summarize_text(src):
    start_time = time()
    combined_text = " ".join(src)

    def summarize(text, max_length):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        # Chunking sentences to fit within model's token limits, skipping long sentences
        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=True)
            if len(sentence_tokens) > 512:
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

        summaries = []
        for chunk in chunks:
            input_ids = tokenizer.encode(
                chunk, return_tensors="pt", add_special_tokens=True
            )
            generated_ids = model.generate(
                input_ids,
                num_beams=4,  # Increased number of beams
                max_length=max_length,
                repetition_penalty=2.5,  # Adjusted penalty
                length_penalty=0.9,
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
        final_summary = " ".join(summaries)
        final_summary = " ".join(
            dict.fromkeys(final_summary.split())
        )  # Remove duplicate words
        return final_summary

    summary = summarize(combined_text, max_length=180)
    end_time = time()
    total_time = end_time - start_time
    print("Total Summarization time:", total_time, "seconds")
    return summary


def translate_text_to_vietnamese(text):
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    translated_sentences = []

    for sentence in sentences:
        if not is_references_section(sentence):
            translated_sentence = model_en_vi.generate(text=text)
            translated_sentences.append(translated_sentence)
        else:
            # Add the sentence as-is if it's part of the references section
            translated_sentences.append(sentence)

    return translated_sentences


def main():
    file_path = "test/news_fr_t.txt"

    text_content = read_file_to_string(file_path)
    text_content_vi = translate_text_to_english(text_content)

    summarize_text(text_content_vi)
    print(summarize_text)

    # base_name = os.path.basename(file_path)
    # name_part = os.path.splitext(base_name)[0]
    # output_file_path = os.path.join("test", name_part + "_t.txt")
    # with open(output_file_path, "w", encoding="utf-8") as file:
    #     for item in combined_text:
    #         file.write(item + "\n")


if __name__ == "__main__":
    print("<utils.py>")
    main()
