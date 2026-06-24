"""Text preprocessing — matches fake_news_eda_improved.ipynb clean_text()."""
import re


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, keep letters/spaces, normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
