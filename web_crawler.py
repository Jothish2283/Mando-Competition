import requests
from bs4 import BeautifulSoup

def crawl_url(url: str, timeout: int = 5) -> str:
    """
    Fetch page at URL and return visible text.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        return f"[Crawl Error] {str(e)}"
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text
