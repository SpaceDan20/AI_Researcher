import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline, BartTokenizer

# Load summarization model and tokenizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


def count_tokens(text):
    # Count tokens in a text string and return the count
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    return inputs["input_ids"].shape[1]


def extract_text(soup):
    # Extract title, subtitles, and main content
    text = []
    for tag in soup.find_all(['h1', 'h2', 'p']):
        if tag.name == 'h1':
            text.append(f"\nTitle: {tag.get_text(strip=True)}\n")
        elif tag.name == 'h2':
            text.append(f"\nSubtitle: {tag.get_text(strip=True)}\n")
        elif tag.name == 'p' and tag.find_parent('footer') is None: # Skip footer text
            text.append(tag.get_text())
    return '\n'.join(text)


def chunk_text(text, max_tokens=400):
    # Break text into chunks if too long for the model
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Try adding the sentence to the current chunk
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        if count_tokens(test_chunk) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = test_chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def summarize_text(text):
    # Summarize text using Hugging Face transformers
    if count_tokens(text) < 30:
        return None  # Skip if text is too short (less than 30 tokens)
    max_summary_length = min(180, int(count_tokens(text) * 0.7))  # 70% of token count, capped at 180
    summary = summarizer(text, max_length=max_summary_length, min_length=30, do_sample=False)
    return summary


def split_into_paragraphs(text, sentences_per_paragraph=4):
    # Split text into sentences to form paragraphs for easier reading
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


url = input("Paste URL: ") # Website to scrape
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
} # Use headers to mimic a real browser request
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
website_text = extract_text(soup) # Extract relevant text from HTML
website_text_chunks = chunk_text(website_text)
chunk_summaries = [] # Create list of summaries for each chunk
for chunk in website_text_chunks:
    chunk_summary = summarize_text(chunk)
    if chunk_summary:
        chunk_summaries.append(chunk_summary)
# Combine chunk summaries into a total summary and split into paragraphs
total_summary = " ".join([s[0]['summary_text'] for s in chunk_summaries if isinstance(s, list)])
total_summary_text = split_into_paragraphs(total_summary, 4)

# Append total summary to an md file
with open("summaries.md", "a", encoding="utf-8") as f:
    f.write(f"\nURL: {url}\n")
    f.write(f"Total Summary:\n{total_summary_text}")
print("\nSummarization complete. Check summaries.md for results.\n")

#TODO: Build a simple UI using Streamlit
#TODO: Polish summarization and sentence chunking
#TODO: Add error handling
