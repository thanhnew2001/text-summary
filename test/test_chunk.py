from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, T5Config

# Load the tokenizer
try:
    tokenizer = T5Tokenizer.from_pretrained(
        "mrm8488/t5-base-finetuned-summarize-news", legacy=False
    )
except Exception as e:
    print(f"Error loading tokenizer: {e}")


def calculate_chunks_0(sentences):
    # Tokenize each sentence and store the tokens
    sentence_tokens = [
        tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences
    ]
    total_tokens = sum(len(tokens) for tokens in sentence_tokens)

    # Check if total tokens exceed the maximum allowed for the whole text
    if total_tokens > 2048:
        return []

    if total_tokens < 512:
        return " ".join(sentences)

    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence, tokens in zip(sentences, sentence_tokens):
        if current_token_count + len(tokens) > 512:
            # If adding this sentence exceeds the max token limit, finalize the current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = []  # Start a new chunk
            current_token_count = 0

        current_chunk.append(sentence)
        current_token_count += len(tokens)

        # Ensure each chunk is at least 256 tokens
        if current_token_count >= 256 and current_token_count <= 512:
            chunks.append(" ".join(current_chunk))
            current_chunk = []  # Start a new chunk
            current_token_count = 0

    # Add the last chunk if any sentences remain
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def calculate_chunks(text):
    sentences = sent_tokenize(text)

    # Tokenize each sentence and store the tokens
    sentence_tokens = [
        tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences
    ]
    total_tokens = sum(len(tokens) for tokens in sentence_tokens)

    # Check if total tokens exceed the maximum allowed for the whole text
    if total_tokens > 2048:
        return []

    if total_tokens < 512:
        return [text]

    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence, tokens in zip(sentences, sentence_tokens):
        if current_token_count + len(tokens) > 512:
            # If adding this sentence exceeds the max token limit, finalize the current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = []  # Start a new chunk
            current_token_count = 0

        current_chunk.append(sentence)
        current_token_count += len(tokens)

        # Ensure each chunk is at least 256 tokens
        if current_token_count >= 256 and current_token_count <= 512:
            chunks.append(" ".join(current_chunk))
            current_chunk = []  # Start a new chunk
            current_token_count = 0

    # Add the last chunk if any sentences remain
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# # Chunking sentences to fit within model's token limits, skipping long sentences
# for sentence in sentences:
#     sentence_tokens = tokenizer.encode(sentence, add_special_tokens=True)
#     if current_length + len(sentence_tokens) <= 512:
#         current_chunk.append(sentence)
#         current_length += len(sentence_tokens)
#     else:
#         chunks.append(" ".join(current_chunk))
#         current_chunk = [sentence]
#         current_length = len(sentence_tokens)
# if current_chunk:
#     chunks.append(" ".join(current_chunk))

# Example usage
text = "The anime industry's growth is also reflected in the increasing number of international collaborations and events. Universal Studios Japan, for example, announced a My Hero Academia event for the first time ever, set to run from March to August 2024, showcasing the theme park's commitment to anime-based attractions[18]. Additionally, the Anime Japan convention continues to be a significant event, drawing over 100,000 guests and serving as a platform for major announcements and exclusive previews[5]. The industry's international appeal is further evidenced by the fact that 44% of American Gen Z respondents watch anime, and over 40% have friends with whom they discuss anime or have read the original manga comics[6]. This demonstrates the formation of a \"Japanese subculture ecosystem\" within American society and indicates the genre's mainstream cultural integration beyond Japan. Looking ahead, the anime market is expected to reach $41.5 billion by 2028, with a compound annual growth rate of 12.3% from 2023 to 2028[7]. This growth is not only a testament to the creativity of anime creators but also to the passion of its fans worldwide. As the industry continues to adapt to global demands, the future of anime holds exciting possibilities for creators, distributors, and fans alike[7]. In summary, the Japanese anime industry is thriving, with significant growth in market value, increased production budgets, and improved working conditions for artists. The global demand for anime content, bolstered by streaming platforms and international events, is driving the industry forward. Despite facing challenges, the anime industry's resilience and adaptability suggest a promising and dynamic future[1][3][5][6][7][18]. The anime industry's growth is also reflected in the increasing number of international collaborations and events. Universal Studios Japan, for example, announced a My Hero Academia event for the first time ever, set to run from March to August 2024, showcasing the theme park's commitment to anime-based attractions[18]. Additionally, the Anime Japan convention continues to be a significant event, drawing over 100,000 guests and serving as a platform for major announcements and exclusive previews[5]. The industry's international appeal is further evidenced by the fact that 44% of American Gen Z respondents watch anime, and over 40% have friends with whom they discuss anime or have read the original manga comics[6]. This demonstrates the formation of a \"Japanese subculture ecosystem\" within American society and indicates the genre's mainstream cultural integration beyond Japan. Looking ahead, the anime market is expected to reach $41.5 billion by 2028, with a compound annual growth rate of 12.3% from 2023 to 2028[7]. This growth is not only a testament to the creativity of anime creators but also to the passion of its fans worldwide. As the industry continues to adapt to global demands, the future of anime holds exciting possibilities for creators, distributors, and fans alike[7]. In summary, the Japanese anime industry is thriving, with significant growth in market value, increased production budgets, and improved working conditions for artists. The global demand for anime content, bolstered by streaming platforms and international events, is driving the industry forward. Despite facing challenges, the anime industry's resilience and adaptability suggest a promising and dynamic future[1][3][5][6][7][18]."

# Test the function with debug output
chunks = calculate_chunks(text)
if chunks:
    print(f"Generated {len(chunks)} chunks.")
else:
    print("Not enough content for chunking or too few sentences.")

for c in chunks:
    print(c)

# can you make sure we chunking properly and max token of 1 chunk is not larger than 512 and smaller than 256
