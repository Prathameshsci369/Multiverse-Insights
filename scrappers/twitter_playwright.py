import json
import asyncio
import re
import os
from datetime import datetime
from urllib.parse import quote
import pdfplumber
import zipfile
from playwright.async_api import async_playwright
import shutil
import warnings
try:
    import pyautogui
    screen_width, screen_height = pyautogui.size()
except:
    screen_width, screen_height = 1920, 1080  # fallback

class TwitterPDFScraper:
    def __init__(self,
                cookies_path="twitter_cookies.json",
                search_query="Maratha",
                pdf_dir="outputs",
                start_date=None,   # NEW
                end_date=None, 
                json_output="tweets_output.json",
                zip_output="twitter_data.zip"):
        self.cookies_path = cookies_path
        self.search_query = search_query
        self.pdf_dir = pdf_dir
        self.json_output = json_output
        self.zip_output_path = zip_output  # renamed
        self.start_date = start_date  # e.g. "2025-07-10"
        self.end_date = end_date 


    async def twitter_search_and_save_pdfs(self):
        os.makedirs(self.pdf_dir, exist_ok=True)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, args=['--start-maximized'])
            context = await browser.new_context(viewport=None, user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/118.0.5993.89 Safari/537.36"
            ))
            # Load cookies
            with open(self.cookies_path, "r") as f:
                cookies = json.load(f)
                for cookie in cookies:
                    if "expirationDate" in cookie:
                        cookie["expires"] = cookie.pop("expirationDate")
                    ss = cookie.get("sameSite")
                    if isinstance(ss, str):
                        ss_lower = ss.lower()
                        if ss_lower == "lax":
                            cookie["sameSite"] = "Lax"
                        elif ss_lower == "strict":
                            cookie["sameSite"] = "Strict"
                        elif ss_lower in ["none", "no_restriction"]:
                            cookie["sameSite"] = "None"
                        else:
                            cookie["sameSite"] = "None"
                    else:
                        cookie["sameSite"] = "Lax"
            await context.add_cookies(cookies)
            page = await context.new_page()
            await page.set_viewport_size({"width": screen_width, "height": screen_height})
            await page.goto("https://x.com/home", timeout=120000, wait_until="domcontentloaded")
            if self.end_date == None:
                self.end_date = datetime.now().strftime("%Y-%m-%d")  # Use current date if not provided

            # Search query
            query_string = self.search_query

            # Add date filters if provided
            if self.start_date :
                query_string += f" since:{self.start_date} until:{self.end_date}"

            encoded_query = quote(query_string)
            search_url = f"https://x.com/search?q={encoded_query}&src=typed_query"


            
            await page.goto(search_url, timeout=200000, wait_until="domcontentloaded")
            await page.wait_for_timeout(10000)
            # Scroll and expand
            for i in range(5):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(10000)
                show_more_buttons = await page.query_selector_all("button")
                for btn in show_more_buttons:
                    try:
                        text = await btn.inner_text()
                        if text.strip().lower() == "show more":
                            await btn.click()
                            await page.wait_for_timeout(2000)
                    except:
                        continue
                await page.wait_for_timeout(8000)
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{self.pdf_dir}/twitter_search_{now}_part{i+1}.pdf"
                await page.pdf(path=filename, format="A4", print_background=True)
                print(f"📄 PDF saved: {filename}")
            await browser.close()

    def extract_text_from_pdf(self, pdf_path):
        all_text = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    all_text += page.extract_text() + "\n"
        return all_text

    from datetime import datetime
    import re

    def extract_tweets_from_text(self, raw_text):
        tweets = []
        lines = raw_text.splitlines()
        all_urls = re.findall(r"https://x\.com/\w+/status/\d+", raw_text)
        url_index = 0
        i = 0

        while i < len(lines):
            if re.match(r'.+@.+·', lines[i]):
                user_info = lines[i]
                content_lines = []
                i += 1

                while i < len(lines) and not re.match(r'^\d+\s[\d\.KkMm]+\s[\d\.KkMm]+\s[\d\.KkMm]+', lines[i]):
                    content_lines.append(lines[i])
                    i += 1

                engagement_line = lines[i] if i < len(lines) else ""
                i += 1

                user_handle_match = re.search(r'(@\w+)', user_info)

                # 🛠️ Updated date regex to capture with OR without year
                date_match = re.search(r'·\s*(\w+\s\d{1,2})(?:,\s*(\d{4}))?', user_info)
                if date_match:
                    month_day = date_match.group(1)     # e.g., "Jul 14"
                    year = date_match.group(2)          # None or "2025"
                    if not year:
                        year = str(datetime.now().year) # Add current year if missing
                    tweet_date = f"{month_day}, {year}"
                else:
                    tweet_date = ""

                likes, retweets, replies, views = "", "", "", ""
                if engagement_line:
                    parts = engagement_line.split()
                    if len(parts) >= 4:
                        replies, retweets, likes, views = parts[:4]

                content = " ".join(content_lines).strip()
                hashtags = re.findall(r"#\w+", content)
                tweet_url = all_urls[url_index] if url_index < len(all_urls) else ""
                url_index += 1

                tweet = {
                    "user_handle": user_handle_match.group(1) if user_handle_match else "",
                    "tweet_date": tweet_date,
                    "content": content,
                    "replies": replies,
                    "retweets": retweets,
                    "likes": likes,
                    "views": views,
                    "tweet_url": tweet_url,
                    "hashtags": hashtags
                }
                tweets.append(tweet)
            else:
                i += 1

        return tweets


    def extract_all_pdfs_and_save_json(self):
        all_tweets = []
        for fname in os.listdir(self.pdf_dir):
            if fname.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_dir, fname)
                print(f"🔍 Extracting from: {pdf_path}")
                text = self.extract_text_from_pdf(pdf_path)
                tweets = self.extract_tweets_from_text(text)
                all_tweets.extend(tweets)
        with open(self.json_output, "w", encoding="utf-8") as f:
            json.dump(all_tweets, f, indent=4, ensure_ascii=False)
        print(f"✅ Extracted {len(all_tweets)} tweets into {self.json_output}")

    def zip_output(self):
        with zipfile.ZipFile(self.zip_output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:

            for fname in os.listdir(self.pdf_dir):
                fpath = os.path.join(self.pdf_dir, fname)
                zipf.write(fpath)
        print(f"🗜 Compressed all output to {self.zip_output_path}")

    def cleanup(self):
        try:
            shutil.rmtree(self.pdf_dir, ignore_errors=True)
            print(f"Directory '{self.pdf_dir}' removed successfully")
        except Exception as e:
            print(f"Error removing directory '{self.pdf_dir}': {e}")

    def run_pipeline(self):
        print("🚀 Running full Twitter scrape → extract → zip pipeline...")
        self.start_date = "2025-06-1"  # Example start date
        #self.end_date = "2024-05-31"  # Example end date
        asyncio.run(self.twitter_search_and_save_pdfs())
        self.extract_all_pdfs_and_save_json()
        self.zip_output()
        self.cleanup()
        print("🎉 All done!")

if __name__ == "__main__":
    scraper = TwitterPDFScraper()
    scraper.run_pipeline()
