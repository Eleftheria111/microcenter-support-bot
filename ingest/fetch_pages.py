import requests
from bs4 import BeautifulSoup
import json, os, time

def fetch_and_parse(url):
    try:
        res = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'lxml')
        for tag in soup(['nav', 'footer', 'script', 'style', 'header']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        title = soup.title.string if soup.title else url
        return {'url': url, 'title': title, 'text': text}
    except Exception as e:
        print(f'Failed {url}: {e}')
        return None

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)
    with open('data/urls.txt') as f:
        urls = [line.strip() for line in f if line.strip()]
    results = []
    for url in urls:
        print(f'Fetching: {url}')
        doc = fetch_and_parse(url)
        if doc: results.append(doc)
        time.sleep(1)
    with open('data/processed/pages.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Done! Saved {len(results)} pages.')