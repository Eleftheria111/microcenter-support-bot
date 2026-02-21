import asyncio
import json
import os
from playwright.async_api import async_playwright

async def scrape_product(page, url):
    try:
        await page.goto(url, timeout=15000, wait_until='domcontentloaded')
        await page.wait_for_timeout(2000)
        
        title = await page.title()
        
        # Get price
        price = ""
        try:
            price = await page.locator('.price, .product-price, [class*="price"]').first.inner_text(timeout=3000)
        except:
            pass
        
        # Get description/main content
        text = await page.evaluate('''() => {
            for (const el of document.querySelectorAll('nav, footer, header, script, style')) {
                el.remove();
            }
            return document.body.innerText;
        }''')
        
        # Build clean document
        content = f"Προϊόν: {title}\n"
        if price:
            content += f"Τιμή: {price}\n"
        content += f"URL: {url}\n"
        content += text[:3000]
        
        print(f"✅ {title[:60]} | {price}")
        return {'url': url, 'title': title, 'text': content}
    
    except Exception as e:
        print(f"❌ Failed {url}: {e}")
        return None

async def main():
    with open('data/product_urls.txt') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Fetching...")
            doc = await scrape_product(page, url)
            if doc:
                results.append(doc)
            await asyncio.sleep(1)
        
        await browser.close()
    
    # Append to existing pages.jsonl
    with open('data/processed/pages.jsonl', 'a') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f'\nΤέλος! Scraped {len(results)} προϊόντα.')

asyncio.run(main())
