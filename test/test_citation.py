import re


def is_references_section(text):
    # Define patterns for detecting citations and reference section titles
    section_titles_pattern = r"(References|Bibliography|Citations|Works Cited)\s*[\n\r]"
    apa_pattern = r"\(.*?\d{4}.*?\)"  # Example: (Author, 2020)
    mla_pattern = r"\w+ \d{4}"  # Example: Author 2020
    harvard_pattern = r"\w+,\s*\w+\.\s*\(\d{4}\)"  # Example: Author, A. (2020)
    chicago_pattern = r"\w+\.\s*\"[^\"]+\"\s*\w+\s*\d{4}"  # Example: Author. "Title" Journal 2020
    simple_citation_pattern = r"([\[\(\{<]\d+[\]\)\}>])\s*(https?://[^\s]+)"  # URL citations with numbers

    # Combine all patterns
    combined_patterns = "|".join([
        section_titles_pattern,
        apa_pattern,
        mla_pattern,
        harvard_pattern,
        chicago_pattern,
        simple_citation_pattern
    ])

    # Search for all matches in the text
    matches = re.finditer(combined_patterns, text, flags=re.IGNORECASE)

    # Calculate the total length of matched text
    matched_length = sum((match.end() - match.start()) for match in matches)

    # Compare the matched text length to the total text length
    # If matched text is more than 50% of total text, return True
    return matched_length > len(text) * 0.5

# Example usage
text_section = """
Main content here...
References
[1] https://www.brookings.edu/articles/an-ideal-public-health-model-vietnams-state-led-preventative-low-cost-response-to-covid-19/
[2] https://thedocs.worldbank.org/en/doc/01092630fc961ea8de0a9382c98ce4a7-0070012022/original/Vietnam-WBG-Part


# Example usage
text_section = """
Main content here...
References
[1] https://www.brookings.edu/articles/an-ideal-public-health-model-vietnams-state-led-preventative-low-cost-response-to-covid-19/
[2] https://thedocs.worldbank.org/en/doc/01092630fc961ea8de0a9382c98ce4a7-0070012022/original/Vietnam-WBG-Partnership-on-COVID19-Preparedness-and-Response.pdf
...
"""

if is_references_section(text_section):
    print("This text section is likely a references or citations section.")
else:
    print("This text section is not identified as a references or citations section.")


# def extract_citations(text):
#     # Improved pattern to capture citation numbers (including brackets) and URLs
#     # This pattern assumes numbers are enclosed by any of the brackets: [], (), {}, or <>
#     pattern = r"([\[\(\{<]\d+[\]\)\}>])\s*(https?://[^\s]+)"

#     # Find all matches in the text
#     matches = re.findall(pattern, text)

#     # Check if matches are correct
#     if not matches:
#         return []

#     # Create a list of strings combining the citation number and URL
#     citations = [f"{num} {url}" for num, url in matches if len(matches[0]) == 2]

#     return citations


# # Example usage
# text_section = """
# Main content here...

# References
# [1] https://www.brookings.edu/articles/an-ideal-public-health-model-vietnams-state-led-preventative-low-cost-response-to-covid-19/
# [2] https://thedocs.worldbank.org/en/doc/01092630fc961ea8de0a9382c98ce4a7-0070012022/original/Vietnam-WBG-Partnership-on-COVID19-Preparedness-and-Response.pdf
# ...
# """

# print("Extracted citations:")
