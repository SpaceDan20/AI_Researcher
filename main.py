import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# Extract title and main content
def extract_text(soup):
    # This function can be customized based on the website structure
    paragraphs = soup.find_all('p')
    text = '\n'.join(p.get_text() for p in paragraphs)
    return text


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
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary


# Website to scrape:
url = input("Paste URL: ")

# Use headers to mimic a real browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

website_text = extract_text(soup)
website_text_chunks = chunk_text(website_text)
chunk_summaries = []
for chunk in website_text_chunks:
    chunk_summary = summarize_text(chunk)
    if chunk_summary:  # Only append if summary is not None
        chunk_summaries.append(chunk_summary)
total_summary = " ".join([s[0]['summary_text'] for s in chunk_summaries if isinstance(s, list)])

# Append total summary to a txt file
with open("summaries.txt", "a", encoding="utf-8") as f:
    f.write(f"\nURL: {url}\n")
    f.write(f"{total_summary}")
print("Summarization complete. Check summaries.txt for results.")

#TODO: Build a simple UI using Streamlit
#TODO: Polish summarization and sentence chunking
#TODO: Add error handling
#TODO: Improve overall summary readability
