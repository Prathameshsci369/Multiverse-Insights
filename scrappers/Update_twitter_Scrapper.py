import json
import asyncio
import re
import os
import logging  # Added for improved logging
from datetime import datetime
from urllib.parse import quote
import random  # Import random for randomized delays
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError  # Import TimeoutError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pyautogui
    screen_width, screen_height = pyautogui.size()
except ImportError as e:
    logger.warning(f"pyautogui not available, using fallback screen size: {e}")
    screen_width, screen_height = 1920, 1080  # fallback

class TwitterScraper:
    """
    A class for scraping tweets and their comments from Twitter (X) using Playwright.

    This scraper performs a search based on a query, collects tweets, and then scrapes comments
    for each tweet concurrently. It handles login via cookies, scrolling to load more content,
    and saves the data to a JSON file.

    Attributes:
        cookies_path (str): Path to the JSON file containing browser cookies for authentication.
        search_query (str): The search query to use for finding tweets.
        start_date (str): Start date for the search in YYYY-MM-DD format (optional).
        end_date (str): End date for the search in YYYY-MM-DD format (optional).
        json_output (str): Name of the output JSON file.
        output_dir (str): Directory to save the output JSON file.
        max_concurrency (int): Maximum number of concurrent comment scraping tasks.
        max_search_scroll_attempts (int): Maximum number of scrolls on the search page.
        max_comment_scroll_attempts (int): Maximum number of scrolls on a tweet's comment section.
        max_comments_per_post (int): Maximum number of comments to scrape per tweet.
        all_tweets (list): List to store all scraped tweet data.
        seen_urls (set): Set to track unique tweet URLs to avoid duplicates.
        logger (logging.Logger): Logger instance for logging messages.
    """

    def __init__(self,
                cookies_path="twitter_cookies.json",
                search_query="ISRO Future space missions",
                start_date=None,   # e.g. "2025-07-10"
                end_date=None,
                json_output="tweets_output.json",
                output_dir="outputs", # ADDED THIS LINE
                max_concurrency=1,
                max_search_scroll_attempts=5, # New: Max scrolls for search results page
                max_comment_scroll_attempts=8, # New: Max scrolls for comments section of a post
                max_comments_per_post=5): # New: Limit for comments per post
        self.cookies_path = cookies_path
        self.search_query = search_query
        self.json_output = json_output
        self.output_dir = output_dir # ADDED THIS LINE
        self.start_date = start_date  # e.g. "2025-07-10"
        self.end_date = end_date
        self.max_concurrency = max_concurrency
        self.all_tweets = []  # List to store all scraped tweet data
        self.seen_urls = set()  # To track unique tweets collected globally
        self.max_search_scroll_attempts = max_search_scroll_attempts
        self.max_comment_scroll_attempts = max_comment_scroll_attempts
        self.max_comments_per_post = max_comments_per_post
        self.logger = logging.getLogger(__name__)  # Logger for this class


    async def extract_tweet_data(self, tweet_element):
        """
        Extracts data from a single tweet element on the page.

        This method parses the tweet's user handle, date, content, hashtags, engagement metrics
        (replies, retweets, likes, views), and URL from the provided tweet element.

        Args:
            tweet_element: The Playwright element representing the tweet.

        Returns:
            dict: A dictionary containing the extracted tweet data, or None if extraction fails.
        """
        try:
            # Extract user handle from the user name element
            user_name_elem = await tweet_element.query_selector('[data-testid="User-Name"]')
            user_name_text = await user_name_elem.inner_text() if user_name_elem else ""
            handle_match = re.search(r'@(\w+)', user_name_text)
            user_handle = f"@{handle_match.group(1)}" if handle_match else ""

            # Extract tweet date from the time element
            time_elem = await tweet_element.query_selector('time')
            datetime_str = await time_elem.get_attribute('datetime') if time_elem else ""
            tweet_date = ""
            if datetime_str:
                try:
                    dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    tweet_date = dt.strftime("%b %d, %Y")
                except ValueError as e:
                    self.logger.warning(f"Error parsing tweet date '{datetime_str}': {e}")
                    tweet_date = "Unknown"

            # Extract tweet content
            content_elem = await tweet_element.query_selector('[data-testid="tweetText"]')
            content = await content_elem.inner_text() if content_elem else ""

            # Extract hashtags from content using regex
            hashtags = re.findall(r"#\w+", content)

            # Extract engagement metrics: replies, retweets, likes, views
            # Check if elements exist before extracting text to avoid errors
            replies_elem = await tweet_element.query_selector('div[data-testid="reply"] span')
            replies = await replies_elem.inner_text() if replies_elem else "0"

            retweet_elem = await tweet_element.query_selector('div[data-testid="retweet"] span')
            retweets = await retweet_elem.inner_text() if retweet_elem else "0"

            like_elem = await tweet_element.query_selector('div[data-testid="like"] span')
            likes = await like_elem.inner_text() if like_elem else "0"

            # Extract views, handling cases where it might not be present
            view_elem = await tweet_element.query_selector('div[data-testid="app-text-transition-container"] span')
            views_text = ""
            if view_elem:
                try:
                    views_text = await view_elem.inner_text()
                except Exception as e:
                    self.logger.warning(f"Error extracting views text: {e}")
                    views_text = "N/A"
            views = views_text.replace(" Views", "").strip() if "Views" in views_text else "N/A"

            # Extract tweet URL from the link element
            url_elem = await tweet_element.query_selector('a[dir="ltr"][role="link"]')
            href = await url_elem.get_attribute('href') if url_elem else ""
            tweet_url = f"https://x.com{href}" if href else ""

            return {
                "user_handle": user_handle,
                "tweet_date": tweet_date,
                "content": content,
                "replies": replies,
                "retweets": retweets,
                "likes": likes,
                "views": views,
                "tweet_url": tweet_url,
                "hashtags": hashtags
            }
        except Exception as e:
            self.logger.error(f"Error extracting tweet data: {e}")
            return None

    async def _check_login_status(self, page):
        """
        Checks if the user is successfully logged in by looking for logged-in specific elements.

        This method looks for the 'Home' button, which is typically present when logged in.

        Args:
            page: The Playwright page object.

        Returns:
            bool: True if logged in, False otherwise.
        """
        try:
            home_button = await page.query_selector('a[aria-label="Home"]')
            if home_button:
                self.logger.info("Login status confirmed via 'Home' button.")
                return True
            else:
                self.logger.warning("Login status NOT confirmed. May need manual intervention or fresh cookies.")
                return False
        except Exception as e:
            self.logger.error(f"Error checking login status: {e}")
            return False

    async def scrape_search_tweets(self, page):
        """
        Scrapes tweets from the search results page based on the search query.

        This method navigates to the Twitter search page, waits for tweets to load,
        and scrolls to collect as many tweets as possible within the specified attempts.
        It uses adaptive scrolling to load more content dynamically.

        Args:
            page: The Playwright page object to use for scraping.
        """
        # Construct the search query string with optional date filters
        query_string = self.search_query
        if self.start_date:
            query_string += f" since:{self.start_date} until:{self.end_date}"

        encoded_query = quote(query_string)
        search_url = f"https://x.com/search?q={encoded_query}"

        self.logger.info(f"Navigating to search URL: {search_url}")
        try:
            await page.goto(search_url, timeout=120000, wait_until="domcontentloaded")
            await asyncio.sleep(5)  # Initial wait for content to load
            await page.wait_for_selector('article[data-testid="tweet"]', timeout=30000, state='visible')  # Wait for first tweet to appear
        except PlaywrightTimeoutError as e:
            self.logger.error(f"Timed out navigating to search page or waiting for initial tweets. URL: {search_url}. Error: {e}")
            return  # Exit if initial load fails
        except Exception as e:
            self.logger.error(f"Error during initial search page navigation: {e}")
            return


        # Initialize variables for scrolling
        last_height = await page.evaluate("document.body.scrollHeight")
        scroll_attempts = 0

        self.logger.info("Starting adaptive scroll and post collection on search page...")
        while scroll_attempts < self.max_search_scroll_attempts:
            # Scroll to the bottom to load more content
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(random.uniform(4, 8))  # Random wait time for dynamic content to load

            # Wait for more tweets to be visible after scrolling
            try:
                await page.wait_for_selector('article[data-testid="tweet"]', state='visible', timeout=10000)
            except PlaywrightTimeoutError as e:
                self.logger.warning(f"Timed out waiting for new tweets after scroll on search page. Error: {e}")
                # This could indicate the end of content or a temporary loading issue

            # Collect all tweet elements currently on the page
            tweet_elements = await page.query_selector_all('[data-testid="tweet"]')
            newly_found_count = 0
            current_urls_on_page = set()
            for te in tweet_elements:
                tweet_data = await self.extract_tweet_data(te)
                if tweet_data and tweet_data["tweet_url"] not in self.seen_urls:
                    self.seen_urls.add(tweet_data["tweet_url"])
                    self.all_tweets.append(tweet_data)
                    newly_found_count += 1
                if tweet_data and tweet_data["tweet_url"]:
                    current_urls_on_page.add(tweet_data["tweet_url"])

            new_height = await page.evaluate("document.body.scrollHeight")

            self.logger.info(f"Scrolled {scroll_attempts + 1}/{self.max_search_scroll_attempts} times. Collected {len(self.all_tweets)} unique posts (new in this scroll: {newly_found_count}).")

            # Check if no new content loaded AND no new tweets found
            if new_height == last_height and newly_found_count == 0:
                self.logger.info("No new content or posts loaded on search page after scrolling. Breaking scroll loop.")
                break  # No more content loading

            last_height = new_height
            scroll_attempts += 1

        self.logger.info(f"Finished initial post collection on search page. Total posts for comment scraping: {len(self.all_tweets)}")


    async def process_comments(self, tweet, context):
        """
        Navigates to an individual tweet page, scrolls to load comments, and scrapes comments data.

        This method opens a new page, navigates to the tweet URL, waits for the main tweet to load,
        then scrolls to load and collect comments. It applies the comment limit and handles retries
        for navigation failures.

        Args:
            tweet (dict): The tweet dictionary containing at least 'tweet_url'.
            context: The Playwright browser context.
        """
        page = await context.new_page()
        await page.set_viewport_size({"width": screen_width, "height": screen_height})

        tweet_url = tweet.get('tweet_url')
        if not tweet_url:
            self.logger.warning("Skipping comment scraping for a tweet with no URL.")
            await page.close()
            return

        self.logger.info(f"Navigating to post: {tweet_url} to scrape comments...")

        max_retries = 3  # Number of retries for navigation/main tweet load
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying navigation/main tweet load for {tweet_url} (Attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(random.uniform(5, 15))  # Longer random sleep before retry

                await page.goto(tweet_url, timeout=120000, wait_until="domcontentloaded")
                await asyncio.sleep(5)  # Initial wait for content to load
                # Increased timeout for waiting for the main tweet
                await page.wait_for_selector('article[data-testid="tweet"]', state='visible', timeout=30000)
                break  # If successful, break retry loop
            except PlaywrightTimeoutError as e:
                self.logger.error(f"Timed out navigating to {tweet_url} or waiting for main tweet (Attempt {attempt + 1}/{max_retries}). Error: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Max retries reached for {tweet_url}. Skipping comments for this post.")
                    await page.close()
                    return
            except Exception as e:
                self.logger.error(f"Error during initial post page navigation for {tweet_url} (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Max retries reached for {tweet_url}. Skipping comments for this post.")
                    await page.close()
                    return

        comments = []  # List to store collected comments
        seen_comment_urls = set()  # To prevent duplicate comments if they load multiple times

        last_height = await page.evaluate("document.body.scrollHeight")
        scroll_attempts = 0

        # Start adaptive scrolling for comments
        while scroll_attempts < self.max_comment_scroll_attempts:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")  # Scroll to bottom
            await asyncio.sleep(random.uniform(3, 6))  # Random wait time for new comments to load

            # Wait for new comment elements to appear
            try:
                # Comments are also tweets, but appear in the reply thread.
                # Increased timeout for waiting for comments after scroll
                await page.wait_for_selector('div[data-testid="cellInnerDiv"] article[data-testid="tweet"]', state='visible', timeout=15000)
            except PlaywrightTimeoutError as e:
                self.logger.warning(f"Timed out waiting for new comments after scroll for {tweet_url}. Error: {e}")
                # This could indicate the end of content or a temporary loading issue

            tweet_elements = await page.query_selector_all('div[data-testid="cellInnerDiv"] article[data-testid="tweet"]')
            newly_found_comments_count = 0

            # Filter by checking for "Replying to" to ensure it's a comment, and skip the main tweet.
            for te in tweet_elements:
                try:
                    reply_indicator = await te.query_selector('div[data-testid="tweetText"] + div div[data-testid="User-Names"] a')
                    if not reply_indicator:  # This is likely the main tweet or a non-reply element
                        continue

                    comment_data = await self.extract_tweet_data(te)
                    # Check for comment_data and if it's a new comment (not already seen)
                    if comment_data and comment_data["tweet_url"] not in seen_comment_urls:
                        comments.append(comment_data)
                        seen_comment_urls.add(comment_data["tweet_url"])
                        newly_found_comments_count += 1
                except Exception as e:
                    self.logger.warning(f"Error extracting individual comment: {e}")
                    pass  # Continue to next comment if one fails

            new_height = await page.evaluate("document.body.scrollHeight")

            # If no new content loaded AND no new comments found, break
            if new_height == last_height and newly_found_comments_count == 0:
                self.logger.info(f"No new content or comments loaded after scroll for {tweet_url}. Breaking comment scroll loop.")
                break

            last_height = new_height
            scroll_attempts += 1

        # Apply the comment limit after all available comments have been collected via scrolling
        if len(comments) > self.max_comments_per_post:
            self.logger.info(f"Limiting comments for {tweet_url} from {len(comments)} to {self.max_comments_per_post}.")
            comments = comments[:self.max_comments_per_post]
        else:
            self.logger.info(f"Collected {len(comments)} comments for {tweet_url} (less than or equal to limit).")

        tweet['comments'] = comments
        self.logger.info(f"Extracted {len(comments)} comments for {tweet_url}")
        await page.close()


    async def scrape_comments_concurrent(self, context):
        """
        Processes comments for all collected tweets concurrently using a semaphore.

        This method creates asynchronous tasks for each tweet to scrape comments,
        limiting concurrency with a semaphore to avoid overwhelming the browser.

        Args:
            context: The Playwright browser context.
        """
        sem = asyncio.Semaphore(self.max_concurrency)

        async def wrapped_process(tweet_item):
            async with sem:
                await self.process_comments(tweet_item, context)

        # Create tasks for all tweets, ensuring only unique ones are processed
        tasks = [wrapped_process(tweet) for tweet in self.all_tweets if tweet.get('tweet_url')]

        await asyncio.gather(*tasks)


    async def scrape_all(self):
        """
        Runs the full Twitter scraping process for posts and comments.

        This method launches a browser, loads cookies, checks login, scrapes tweets from search,
        then scrapes comments concurrently, and finally saves the data to a JSON file.
        """
        self.logger.info("Running full Twitter scrape for posts and comments...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, args=['--start-maximized'])
            context = await browser.new_context(viewport=None, user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/118.0.5993.89 Safari/537.36"
            ))

            # Load cookies
            try:
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
                            else:  # Default to Lax if unrecognizable for Playwright
                                cookie["sameSite"] = "Lax"
                        else:
                            cookie["sameSite"] = "Lax"  # Default

                        # Remove potentially problematic fields
                        cookie.pop("partitionKey", None)
                        cookie.pop("firstPartyDomain", None)
                        cookie.pop("storeId", None)

                await context.add_cookies(cookies)
                self.logger.info(f"Loaded cookies from {self.cookies_path}")
            except FileNotFoundError as e:
                self.logger.error(f"Cookies file not found at {self.cookies_path}. Please ensure it exists. Error: {e}")
                await browser.close()
                return
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding JSON from {self.cookies_path}. Ensure it's valid JSON. Error: {e}")
                await browser.close()
                return
            except Exception as e:
                self.logger.error(f"An error occurred loading or processing cookies: {e}")
                await browser.close()
                return

            page = await context.new_page()
            await page.set_viewport_size({"width": screen_width, "height": screen_height})

            # Go to home and check login status
            await page.goto("https://x.com/home", timeout=120000, wait_until="domcontentloaded")
            await asyncio.sleep(5)  # Give time to load
            if not await self._check_login_status(page):
                self.logger.error("Login failed or cookies are invalid. Please update 'twitter_cookies.json'.")
                await browser.close()
                return

            await self.scrape_search_tweets(page)

            # Now, scrape comments concurrently for all collected tweets
            self.logger.info(f"Starting concurrent comment scraping for {len(self.all_tweets)} posts...")
            await self.scrape_comments_concurrent(context)

            await browser.close()

        # Save all scraped data to a JSON file
        output_filepath = os.path.join(self.output_dir, self.json_output)
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(self.all_tweets, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Extracted {len(self.all_tweets)} tweets with comments into {output_filepath}")
        except Exception as e:
            self.logger.error(f"Error saving data to {output_filepath}: {e}")
            return
        self.logger.info("All Twitter scraping done!")

    def run_pipeline(self):
        """
        Runs the full Twitter scraping pipeline.

        This method initiates the asynchronous scraping process and handles the overall execution.
        """
        self.logger.info("Running full Twitter scrape → extract → json pipeline...")
        # You can set start_date and end_date here if you want to override __init__ defaults
        # self.start_date = "2024-06-01"
        # self.end_date = "2025-08-28"
        try:
            asyncio.run(self.scrape_all())
            self.logger.info("All done!")
        except Exception as e:
            self.logger.error(f"Error running pipeline: {e}")

if __name__ == "__main__":
    """
    Main entry point for the Twitter scraper script.

    This block sets up the scraper with the desired parameters and runs the pipeline.
    """
    # Ensure 'outputs' directory exists for JSON output
    output_dir = "outputs"
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {e}")
        exit(1)

    # Initialize the scraper with parameters
    scraper = TwitterScraper(
        search_query="India China relations",  # Your search query
        cookies_path=os.path.join("twitter_cookies.json"),  # Path to cookies file
        json_output="tweets_output.json",  # Output file for scraped data
        # start_date="2025-06-01",  # Optional: Specify start date (YYYY-MM-DD)
        # end_date="2025-08-28",  # Optional: Specify end date (YYYY-MM-DD)
        # max_search_scroll_attempts=20,  # Increased for potentially more posts
        # max_comment_scroll_attempts=15,  # Increased for potentially more comments
        # max_comments_per_post=100  # Adjust this limit as needed
    )
    scraper.run_pipeline()

