**Multiverse Insights Data Scraping Toolkit Documentation**

This documentation covers three modules for scraping and processing content from Twitter (X), Telegram, and Reddit. Each module leverages modern libraries, asynchronous techniques, and error‑handling strategies to efficiently collect and store social media data.

---

## 1. Twitter PDF Scraper (`twitter_playwright.py`)

**Key Technologies:**

* **Playwright (async)**: Headless browser automation for dynamic content loading and PDF generation.
* **PDFPlumber**: Extraction of text from generated PDFs.
* **PyAutoGUI (optional)**: Automatic screen size detection.

**Workflow & Techniques:**

1. **Browser Context & Cookies**: Reuses authenticated cookies for stable sessions. Normalizes `sameSite` attribute.
2. **Dynamic Scrolling & Interaction**: Scroll loop with `page.wait_for_timeout()` and "Show more" button clicks to load tweets.
3. You can add the clicking the comments button and take there comments .
4. optional `since:YYYY-MM-DD until:YYYY-MM-DD` in search query.
5. **PDF Generation**: Saves snapshots in A4 format, segmented parts to avoid timeouts.
6. **Text Extraction**: Opens each PDF with `pdfplumber`, extracts text, then applies regex to capture tweets, dates, engagement, and URLs.
7. **Pipeline**: High-level `run_pipeline()` to orchestrate search → PDF save → extract → JSON save → ZIP → cleanup.

**Error Handling & Performance:**

* **Timeouts**: High `timeout` values for network operations.
* **Warnings Filter**: Suppresses PDF parsing warnings.
* **Batching**: Breaks scrolls into chunks for stability.

---

## 2. Telegram Scraper (`telegram_scraper.py`)

**Key Technologies:**

* **Telethon (async)**: Telegram API client for bot interaction, channel join/leave, and history retrieval.
* **Asyncio**: Non‑blocking delays and parallel tasks.
* **Regex & Caching**: Keyword matching, flood‑wait cooldown caching.

**Workflow & Techniques:**

1. **Bot Pagination**: Queries via SearchBot(use can use different bots/sevices), caches last bot messages, clicks “Next” buttons with limited retries.
2. **Link Extraction**: Parses `message.entities` for URLs, writes to `scraped_links.txt`.
3. **Channel Scraping without Joining**: Uses `get_entity` and `GetHistoryRequest` to read public channels. Avoids join when over limit.
4. **Join/Leave Strategy**: Async delays after join/leave to mimic human behavior; optional throttle in config.
5. **FloodWait Handling**: Caches usernames with `FloodWaitError`, waits inline if short (<60s), or loops with 2s sleeps until resume time.
6. **Keyword Filtering**: Precompiles word‑boundary regex for multi‑word keywords. Limits per‑channel message count.
7. **Data Persistence**: JSON output for channel messages, periodic saves to prevent data loss.

**Best Practices:**

* **Rate Limiting**: Configurable `REQUEST_DELAY`, flood‑wait cache to prevent rapid retries.
* **Graceful Degradation**: Skips private/admin‑only channels, logs errors without stopping the pipeline.
* **Resource Cleanup**: Leaves channels and closes client contexts.

---

## 3. Reddit Scraper (`reddit.py`)

**Key Technologies:**

* **PRAW**: Reddit API wrapper for authenticated scraping.
* **Google Gemini API**: ML-driven query reduction to generate concise search terms.
* **Requests**: HTTP calls to Gemini endpoint.
* **Rate Limiting**: Simple `time.sleep()` between requests.

**Workflow & Techniques:**

1. **Query Refinement**: Sends user prompt to Gemini API for keyword‑based query generation, handles 400/403/404 responses.
2. **Authentication & Error Checks**: Validates credentials; catches `prawcore.exceptions` and general PRAW errors.
3. **Subreddit Search**: ML‑refined query → `subreddits.search()` → list of target subreddits.
4. **Post Fetching**: Retrieves hot or search results; filters posts with non‑empty `selftext`.
5. **Comment Extraction**: Grabs top comments and subcomments by score; structures nested dictionaries.
6. **Topic Extraction (optional)**: Tokenizes text to extract most frequent topics.
7. **Data Output**: Saves JSON results, logs summary.

**Advanced Techniques:**

* **Dynamic Search Limit**: Fetches 2× requested limit, stops when desired count reached.
* **Duplicate Detection**: Tracks seen URLs to avoid repeats.
* **Feedback Loop**: Allows user to refine Gemini prompt if unsatisfied.

---

## General Best Practices & Techniques

1. **Asynchronous Programming**: Leverage `asyncio` and async client libraries to maximize throughput.
2. **Rate‑Limit Awareness**: Implement delays, cooldown caches, and skip logic to handle API limits and avoid blocks.
3. **Structured Logging**: Use Python’s `logging` module with file and console handlers, consistent formats.
4. **Configuration Management**: Centralize constants in a config class (`TelegramConfig`), easily tunable for different scenarios.
5. **Modular Design**: Keep scraper logic separated per platform, reusable extraction methods.
6. **Error Recovery**: Wrap API calls in try/except blocks, log and skip rather than crash.
7. **Data Persistence**: Periodic saves to JSON/text files, safe cleanup of temporary artifacts.
8. **ML Integration**: Use LLM-driven query generation to improve search relevance on platforms with limited search syntax.

---

By combining these modules and techniques, this toolkit provides a robust foundation for scraping, processing, and analyzing social media data across multiple platforms. Adjust configurations (delays, limits, keywords) to suit specific research or production requirements.
