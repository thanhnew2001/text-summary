# import re


# def clean_special_characters(text):
#     pattern = r"\[\d+(?:\]\[\d+)*\](?=[.,;:]?(?:\s|$))"
#     return re.sub(pattern, "", text)


# def summarize_text(text_list):
#     # This is a placeholder function; you should replace it with your actual text summarization logic
#     return " ".join(text_list)  # Summarize by concatenating for demonstration


# # Assuming txt_en is your list of strings
# txt_en = [
#     "Looking ahead, the anime market is expected to reach $41.5 billion by 2028, with a compound annual growth rate of 12.3% from 2023 to 2028[7].",
#     "This growth is not only a testament to the creativity of anime creators but also to the passion of its fans worldwide[7].",
#     "In summary, the Japanese anime industry is thriving, with significant growth in market value[1][3][5][6][7][18].",
# ]

# # Clean each string in the list from IEEE references and summarize
# cleaned_texts = [clean_special_characters(text) for text in txt_en]
# summarized_text = summarize_text(cleaned_texts)

# print(summarized_text)


import re


def clean_text(text):
    # Replace multiple "n" characters with a single "n"
    cleaned_text = re.sub(r"n+", "", text)
    return cleaned_text


# Example text with repeated "n" characters
text = "the ongoing conflict between Russia and Ukraine has escalated significantly, with recent developments highlighting the increase in military operations and wider geopolitical impacts. Germany has announced it will hand over Patriot air defense and missiles to Ukraine to support defense efforts in the region. The United States has also imposed sanctions on Russia, aimed at undermining the ability to sustain war effort. Notably, the conflict is the fiercest on European soil for over 70 years.nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn the Russia-Ukraine war continued to escalate, with significant military, humanitarian and geopolitical consequences. The re-election of President Vladimir Putin is considered positive for an increasingly close alliance with Beijing. However, the resolution of the conflict remains uncertain, with ongoing military actions, strategic targeting of infrastructure, and the complex interaction of international relationships that shape the course of events. Notably, the refugee crisis has been described as worst in Europe since World War II. Notably, the refugee crisis has also caused millions of Ukrainians to leave their"

# Clean the text
cleaned_text = clean_text(text)

print(cleaned_text)
