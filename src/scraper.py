import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TIMEOUT = 10


def scrape_job_text(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ValueError("URL must start with http:// or https://")

    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ConnectionError("Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Could not fetch URL: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles, nav, footer noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Try to find main content containers common on job sites
    candidates = soup.find_all(
        ["article", "section", "div"],
        class_=re.compile(
            r"job|description|posting|detail|content|body", re.IGNORECASE
        ),
    )
    if candidates:
        text = " ".join(c.get_text(separator=" ", strip=True) for c in candidates[:3])
    else:
        text = soup.get_text(separator=" ", strip=True)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 50:
        raise ValueError("Could not extract enough text from the provided URL.")

    return text
