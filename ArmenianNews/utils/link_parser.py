import requests
from bs4 import BeautifulSoup
import json
import csv
from typing import List, Dict

class LinkParser:
    def fetch_and_extract(url: str) -> Dict:
        USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        HEADERS = {"User-Agent": USER_AGENT}
        try:
            host = requests.utils.urlparse(url).netloc.lower()
        except Exception:
            return {'url': url, 'error': 'invalid url', 'title': '', 'text': ''}

        def _extract_api(site: str, api_base: str) -> Dict:
            path = requests.utils.urlparse(url).path
            last = path.rstrip('/').split('/')[-1]
            article_id = ''.join(ch for ch in last if ch.isdigit())
            parts = [p for p in path.split('/') if p]
            lang = parts[0] if parts else ('ru' if site == 'tert' else 'am')
            api_url = f'{api_base}/{lang}/{article_id}'
            try:
                r = requests.get(api_url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                resp = r.json()
                news = resp.get('data', {}).get('newsitem') or resp.get('data', {}).get('news') or {}
                title = news.get('title') or news.get('heading') or ''
                date = news.get('date') or ''
                text_html = news.get('body') or news.get('text') or news.get('description') or ''
                if text_html:
                    inner = BeautifulSoup(text_html, 'lxml')
                    paragraphs = [p.get_text(separator=' ', strip=True) for p in inner.find_all('p')]
                    text = ' '.join([p for p in paragraphs if p])
                else:
                    text = news.get('content') or ''
                return {'url': url, 'title': title, 'text': text, 'date': date}
            except Exception as e:
                return {'url': url, 'error': f'api_error: {e}', 'title': '', 'text': '', 'date': ''}

        if 'tert.am' in host:
            return _extract_api('tert', 'https://api.tert.am/api/v1/newsdetails/tert')
        if 'panorama.am' in host or 'panorama' in host:
            return _extract_api('panorama', 'https://api.panorama.am/api/v1/newsdetails/panorama')

        # hetq.am - HTML pages only: extract from .news-content-block and .block-header
        if 'hetq.am' in host or 'hetq' in host:
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                soup = BeautifulSoup(r.content, 'lxml')
                title = ''
                h = soup.select_one('h1.block-header') or soup.find('h1')
                if h:
                    title = h.get_text(strip=True)
                content = soup.select_one('div.news-content-block')
                paragraphs = []
                if content:
                    # remove comment blocks and report block so their text isn't included
                    for sel in ('.comments', '.comments.shadow', '.report-article'):
                        for node in content.select(sel):
                            node.decompose()
                    for p in content.find_all(['p', 'div']):
                        # skip script/style or empty nodes
                        if p.name in ('script', 'style'):
                            continue
                        text = p.get_text(separator=' ', strip=True)
                        if text:
                            paragraphs.append(text)
                text = ' '.join(paragraphs)
                # extract date: prefer <time datetime=> elements
                date = ''
                tnode = soup.find('time', attrs={'datetime': True}) or soup.select_one('time.cw-relative-date') or soup.find('time')
                if tnode:
                    date = tnode.get('datetime') or tnode.get_text(strip=True)
                return {'url': url, 'title': title, 'text': text, 'date': date}
            except Exception as e:
                return {'url': url, 'error': f'hetq_html_error: {e}', 'title': '', 'text': ''}

        # 1in.am - HTML parsing for article pages
        if '1in.am' in host or host.endswith('.1in.am'):
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                soup = BeautifulSoup(r.content, 'lxml')
                # title
                title = ''
                h = soup.select_one('h1.single_post_title_standard') or soup.find('h1')
                if h:
                    title = h.get_text(strip=True)

                # main article content: look for common containers
                content = soup.select_one('div.single-post-content') or soup.select_one('div.article-content') or soup.select_one('div.post-content')
                paragraphs = []
                if content:
                    # remove known UI blocks that may contain ads/forms
                    for sel in ('.news-photo', '.ad', '.banner', '.adriverBanner', '.category_block_14', '.item_image', 'iframe'):
                        for node in content.select(sel):
                            node.decompose()
                    # collect text from paragraphs and direct div children
                    for el in content.find_all(['p', 'div']):
                        if el.name in ('script', 'style'):
                            continue
                        txt = el.get_text(separator=' ', strip=True)
                        if txt:
                            paragraphs.append(txt)
                text = ' '.join(paragraphs)

                # date: try post_meta date containers or time tags
                date = ''
                dnode = soup.select_one('div.post_item_date') or soup.select_one('div.item_date') or soup.find('time')
                if dnode:
                    date = dnode.get('datetime') or dnode.get_text(strip=True)

                return {'url': url, 'title': title, 'text': text, 'date': date}
            except Exception as e:
                return {'url': url, 'error': f'1in_html_error: {e}', 'title': '', 'text': '', 'date': ''}

        return {'url': url, 'error': 'unsupported host', 'title': '', 'text': '', 'date': ''}

    def load_urls_from_file(path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def save_results_json(results: List[Dict], outpath: str):
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def save_results_csv(results: List[Dict], outpath: str):
        keys = ['url', 'title', 'text', 'date_time']
        with open(outpath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, '') for k in keys})


    def parse_articles(urls: List[str] = None, input_file: str = None,
                       out_json: str = 'articles.json', out_csv: str = 'articles.csv') -> List[Dict]:
        """Основная функция для извлечения статей из URL"""
        # Сбор URL из разных источников
        extracted_urls = []

        if input_file:
            extracted_urls.extend(LinkParser.load_urls_from_file(input_file))

        if urls:
            extracted_urls.extend(urls)

        # Очистка и проверка URL
        extracted_urls = [u for u in extracted_urls if u]
        if not extracted_urls:
            print('No URLs provided. Provide either urls list or input_file')
            return []

        # Парсинг каждой статьи
        results = []
        for url in extracted_urls:
            print(f'Fetching: {url}')
            res = LinkParser.fetch_and_extract(url)
            results.append(res)

        # Сохранение результатов
        if out_json:
            LinkParser.save_results_json(results, out_json)
        if out_csv:
            LinkParser.save_results_csv(results, out_csv)

        print(f'Done. Parsed {len(results)} articles')
        if out_json:
            print(f'JSON saved to: {out_json}')
        if out_csv:
            print(f'CSV saved to: {out_csv}')

        return results