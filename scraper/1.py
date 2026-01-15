from playwright.sync_api import sync_playwright
import pandas as pd
import time

URL = "https://pk.khaadi.com/ready-to-wear/"

data = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(URL, timeout=60000)

    # Scroll to load all products
    for _ in range(10):
        page.mouse.wheel(0, 5000)
        time.sleep(1.5)

    products = page.query_selector_all("div.product-item")

    for product in products:
        try:
            title = product.query_selector("a.product-item__title").inner_text()
            price = product.query_selector("span.price").inner_text()
            link = product.query_selector("a.product-item__title").get_attribute("href")
            image = product.query_selector("img").get_attribute("src")

            data.append({
                "title": title,
                "price": price,
                "product_url": "https://pk.khaadi.com" + link,
                "image_url": image
            })
        except Exception:
            continue

    browser.close()

df = pd.DataFrame(data)
df.to_csv("khaadi_ready_to_wear.csv", index=False)

print(f"Scraped {len(df)} products")
