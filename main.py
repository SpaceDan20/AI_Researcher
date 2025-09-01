import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# Extract title and main content
def extract_text(soup):
    text = []
    for tag in soup.find_all(['h1', 'h2', 'p']):
        if tag.name == 'h1':
            text.append(f"\nTitle: {tag.get_text(strip=True)}\n")
        elif tag.name == 'h2':
            text.append(f"\nSubtitle: {tag.get_text(strip=True)}\n")
        elif tag.name == 'p' and tag.find_parent('footer') is None:
            text.append(tag.get_text())
    return '\n'.join(text)


# Break text into chunks if too long for the model
def chunk_text(text, max_tokens=300):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    for sentence in sentences:
        sentence_tokens = sentence.split()
        # If a single sentence is longer than max_tokens, split it
        while len(sentence_tokens) > max_tokens:
            chunks.append(' '.join(sentence_tokens[:max_tokens]))
            sentence_tokens = sentence_tokens[max_tokens:]
        # If adding this sentence exceeds max_tokens, start a new chunk
        if len(current_chunk) + len(sentence_tokens) > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = sentence_tokens
        else:
            current_chunk += sentence_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# Summarize text using Hugging Face transformers
def summarize_text(text):
    if not text.strip() or len(text.split()) < 10:
        return None  # Skip empty or very short text
    max_summary_length = min(150, int(len(text.split()) * 0.7))  # 70% of text length, capped at 150
    summary = summarizer(text, max_length=max_summary_length, min_length=30, do_sample=False)
    return summary


def split_into_paragraphs(text, sentences_per_paragraph=4):
    # Split text into sentences to form paragraphs
    sentences = re.split(r'(?<=[.!?]) +', text)
    paragraphs = []
    i = 0
    while i < len(sentences):
        # If this is the last sentence and it's alone, append it to the last paragraph
        if i == len(sentences) - 1 and paragraphs:
            paragraphs[-1] += ' ' + sentences[i].strip()
            break
        paragraph = ' '.join(sentences[i:i+sentences_per_paragraph]).strip()
        if paragraph:
            paragraphs.append(paragraph)
        i += sentences_per_paragraph
    # Join paragraphs
    return '\n\n'.join(p for p in paragraphs)


# Website to scrape:
url = input("Paste URL: ")

# Use headers to mimic a real browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}
# Grab HTML from URL
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
# Extract relevant text from HTML
website_text = extract_text(soup)
website_text_chunks = chunk_text(website_text)
chunk_summaries = []
for chunk in website_text_chunks:
    chunk_summary = summarize_text(chunk)
    if chunk_summary:  # Only append if summary is not None
        chunk_summaries.append(chunk_summary)
# Combine chunk summaries into a final summary
total_summary = " ".join([s[0]['summary_text'] for s in chunk_summaries if isinstance(s, list)])
total_summary_text = split_into_paragraphs(total_summary, 4)

# Append total summary to an md file
with open("summaries.md", "a", encoding="utf-8") as f:
    f.write(f"\nURL: {url}\n")
    f.write(f"Total Summary:\n{total_summary_text}")
print("\nSummarization complete. Check summaries.md for results.")

#TODO: Build a simple UI using Streamlit
#TODO: Polish summarization and sentence chunking
#TODO: Add error handling
