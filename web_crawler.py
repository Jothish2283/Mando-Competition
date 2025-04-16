import requests
from bs4 import BeautifulSoup

def crawl_url(url, timeout=5):
    """
    Fetch and extract visible text from the URL.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    
    soup = BeautifulSoup(response.text, "html.parser")
    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text
