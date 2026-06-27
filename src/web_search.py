import requests
from bs4 import BeautifulSoup





def get_webpage_text(url):
    try:
        html = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        ).text

        soup = BeautifulSoup(html, "html.parser")

        return soup.get_text(" ", strip=True)

    except Exception as e:
        print("Webpage Error:", e)
        return ""