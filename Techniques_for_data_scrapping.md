

---

## ✅ Technologies That Simplify Scraping

### 1. **Asynchronous Programming with** `asyncio`

* Used in Telegram and Playwright scraping to perform concurrent tasks without blocking.
* Improves scraping throughput by avoiding wait times on I/O-bound tasks (e.g., API requests, delays).

### 2. **Playwright for PDF Extraction**

* **Use Case:** Scraping content from dynamically rendered Twitter and Reddit pages. Instead adding the css selectors you can direclty convert page into the PDF.
* **Method:** Automate scrolling and button clicks to load full post/tweet history. Then, capture snapshots as PDFs.
* **Advantages:** Handles JavaScript-heavy sites where HTML-based parsers fail.

### 3. **PDFPlumber + Regex for Structured Data Extraction**

* Extracts raw text from PDFs.
* Regex patterns are used to extract:

  * ✅ Tweet/post content
  * ✅ Dates
  * ✅ Likes, retweets, scores
  * ✅ Comments and user info
  * ✅ Post/tweet links

#### Example Regex Breakdown:

```python
r"https://twitter.com/\w+/status/\d+"
```

* Extracts tweet links.
* `\w+`: Twitter handle
* `\d+`: Tweet ID

```python
r"Score: (\d+)"
```

* Captures Reddit post scores

```python
r"Posted by u/\w+"
```

* Finds Reddit user handles

```python
r"(\d{1,2} \w+ \d{4})"
```

* Extracts human-readable dates like "14 July 2025"

```python
r"(\d+ points|\d+ upvotes)"
```

* Scores from Reddit in different styles

```python
r"Comment by u/\w+: .*?\n"
```

* Reddit comment matching

```python
r"r/\w+"
```

* Subreddit name references

Regex allows developers to convert unstructured PDF text into structured dictionaries ready for analysis.

### 4. **FloodWait Cooldown Cache (Telegram)**

* Caches usernames when Telegram triggers flood control.
* Automatically waits in a loop (2s interval) until the required cooldown time has passed.
* Prevents repeated failed requests.

### 5. **AI‑Driven Query Reduction (Reddit & YouTube)**

* Uses Gemini API to simplify user prompts into precise keyword-based queries.
* Improves search targeting and result relevance.

---

## 🔍 Reddit Pipelines (With & Without Login)

### A. Using Reddit API (PRAW)

* Accepts user prompt → Gemini refines it → passes to `praw.Reddit` client.
* Retrieves top posts and comments using `.subreddit().search()` or `.hot()`
* Extracted fields:

  * `title`, `selftext`, `url`, `score`, `num_comments`, `created_utc`
  * `comments`: top-level and nested replies



---

### B. Alternate Reddit Scraping via Playwright (No API Key)

* Launches browser with **Playwright**
* Scrolls and loads dynamic Reddit content
* Saves the rendered page as **PDF**
* Extracts structured data using **PDFPlumber + Regex**:

  * ✅ Scores
  * ✅ Usernames
  * ✅ Comments
  * ✅ Post links
  * ✅ Dates and timestamps
  * ✅ Post content
  * ✅ Subreddit mentions

#### 📌 Sample Regex Patterns for Reddit PDF Parsing

```python
r"r/\w+"  # Subreddit mentions like r/Python
r"Score: \d+"  # Post score e.g., Score: 823
r"Posted by u/\w+"  # Username e.g., Posted by u/Redditor123
r"(\d{1,2} \w+ \d{4})"  # Date format like 14 July 2025
r"Comment by u/\w+: .*?\n"  # Comment line with user and text
```

* Converts matched text into **JSON** and **CSV** outputs
* Helps bypass **Reddit API key limits**
* Fully compatible with **headless scraping environments**
* No login or authentication required

This method is extremely useful when you're scraping public Reddit threads without using the official API, or when you need to extract **archival snapshots** and **offline analysis** from PDF exports.




---

## 📲 Telegram Scraping Pipeline

1. Accepts channel/group link
2. Joins or resolves entity (unless flood-wait triggered)
3. Collects recent messages using `GetHistoryRequest`
4. Filters messages using word-by-word keyword matching:

   * `re.compile(r'\\b(' + '|'.join(keywords) + r')\\b', re.IGNORECASE)`
   * Ensures matches are exact words, not substrings
5. Captures sender ID and message content
6. Saves results in structured format (e.g., JSON)

**Flood Wait Handling:**

* When a FloodWaitError is triggered:

  * Log and skip until the exact cooldown timestamp
  * Maintain a cooldown cache to skip further resolution attempts during the wait

---

## 📺 YouTube Scraping Pipeline

### Overview

1. **User Prompt** → processed with Gemini → optimal search query.
2. **Search via YouTube API** → retrieve top video IDs.
3. **Collect Metadata**:

   * `title`, `channel`, `likes`, `views`, `comments`, `description`

### 🔠 Transcript Handling

#### A. Default Method

* Use `youtube-transcript-api` to fetch captions.
* Fails when:

  * Transcripts are disabled
  * Videos have no audio

#### B. Whisper‑Based Fallback (OpenAI Whisper Small/Medium)

* Download video audio using `yt_dlp` or `pytube`.
* Pass audio to `whisper` model:

  * Supports multiple languages
  * Can auto‑translate audio to a target language
* Reliable for extracting text even from muted or no-caption videos.

#### C. Lightweight Translation (Optional)

* Use HuggingFace models for:

  * Text-to-text translation of non-English captions
  * Language detection and switching

### Final Output

* Store extracted info to:

  * `video_data.json`
  * `video_transcripts.csv`



# 📸 Instagram Scraping Techniques

Efficient reel scraping using official API and a robust Playwright-based alternative. Whisper automatically extracts and transcribes audio from video — no manual audio handling needed.

---

## 1️⃣ Official Method: Instagram Graph API

- Requires App ID, App Secret, Access Token (Meta Developer Console)
- Accesses media metadata (captions, timestamps, media_type)
- No access to comments, reels audio, or transcripts
- Rate-limited and approval required

---

## 2️⃣ Alternative: Playwright + Whisper

###  Tech Stack & Pipeline:

- `Gemini`: Refines user search query
- `Playwright`: Logs in via cookies, automates Instagram search
- Scrolls and captures result page as PDF
- `PDFPlumber + Regex`: Extracts reel URLs from PDF
- `Instaloader` or `yt_dlp`: Downloads reels, captions, likes, comments
- `Whisper`: Accepts full reel video, auto-extracts and transcribes audio (multilingual + translation)
- Results saved as structured `JSON` for further processing

---

## ✅ Advantages:

- No API keys or rate limits
- Works with transcript-disabled reels
- Auto-translates reel speech to target language
- Bypasses caption limitations with full audio transcription
- Efficient for large-scale, language-diverse scraping


---
---

## 🧠 Developer Tips

* Always **compile regex** with `re.IGNORECASE` and `re.DOTALL` when needed.
* Pre-validate PDF text via `.extract_text()` before parsing.
* Use `re.findall()` for lists (e.g., links, hashtags) and `re.search()` for single match extraction.

```python
# Extract hashtags
re.findall(r"#\w+", text)
```

* Maintain modular pipeline: `input → refine → extract → transform → save`
* Cache failures and cooldowns to improve retry logic
* Store intermediate data to reduce re-runs

By combining async scraping, browser automation, AI-powered query optimization, PDF extraction, regex parsing, and fallback techniques like Whisper, we enable scalable and language-agnostic social media data mining across platforms.
