
---

## ✅ Technologies That Simplify Scraping

### 1. \*\* Asynchronous Programming with \*\***`asyncio` **

* Used in Telegram and Playwright scraping to perform concurrent tasks without blocking.
* Improves scraping throughput by avoiding wait times on I/O-bound tasks (e.g., API requests, delays).

### 2. **Playwright for PDF Extraction**

* **Use Case:** Scraping content from dynamically rendered Twitter pages.
* **Method:** Automate scrolling and button clicks to load full tweet history. Then, capture snapshots as PDFs.
* **Advantages:** Handles JavaScript-heavy sites where HTML-based parsers fail.

### 3. **PDFPlumber + Regex for Structured Data Extraction**

* Extracts raw text from PDFs.
* Regex patterns are used to extract:

  * ✅ Tweet content
  * ✅ Dates
  * ✅ Retweet, like, and reply counts
  * ✅ Tweet links
  * ✅ Comments 

#### Example Regex Breakdown:

```python
r"https://twitter.com/\w+/status/\d+"
```

* **Pattern:** Extracts tweet links.
* `\w+`: Matches Twitter handle
* `\d+`: Matches the tweet ID

```python
r"(\d{1,2} \w+ \d{4})"
```

* **Pattern:** Extracts dates like "14 July 2025"
* `\d{1,2}`: Day
* `\w+`: Month (word)
* `\d{4}`: Year

Regex allows developers to convert unstructured PDF text into structured dictionaries ready for analysis.

### 4. **FloodWait Cooldown Cache (Telegram)**

* Caches usernames when Telegram triggers flood control.
* Automatically waits in a loop (2s interval) until the required cooldown time has passed.
* Prevents repeated failed requests.

### 5. **AI‑Driven Query Reduction (Reddit)**

* Uses Gemini API to simplify complex search prompts into optimized keyword-based queries.
* Reduces noise and improves subreddit targeting.

---

## 🔍 Updated Reddit Pipeline (No Login Required)

Reddit allows public data access for most posts and comments without login.

### New Workflow:

1. **Accepts User Prompt** → passes to Gemini → generates optimized keyword query.
2. **Search Subreddits via PRAW**: Uses `.search()` or `.subreddit().hot()` with keywords.
3. **Filter Relevant Posts**:

   * Only include posts with body (`selftext`) content.
   * Fetch top `n` posts.
4. **Extract Top-Level Comments**:

   * Retrieves top comments and replies with metadata (score, author, depth).
5. **Save to JSON**:

   * Final structured output for posts + comments.

### Extracted Data Fields:

* `title`
* `url`
* `score`
* `selftext`
* `created_utc`
* `num_comments`
* `comments` (nested dict with replies)

---

## 🆕 Alternate Reddit Extraction Using Playwright (No API Key)

To bypass API limits and access dynamic Reddit content directly from the browser:

### Steps:

1. **Launch Headless Chromium using Playwright**

   * Navigate to target subreddit or search page.
   * Emulate scroll behavior to load content.

2. **Capture Dynamic Page as PDF**

   * Full-page PDF generation in A4 segments.

3. **Extract Content with PDFPlumber**

   * Parses textual structure into raw string data.

4. **Apply Regex Parsing**

```python
r"Score: (\d+)"
```

→ Captures score (upvotes)

```python
r"Posted by u/\w+"
```

→ Extracts post authors

```python
r"https://www.reddit.com/r/[\w_]+/comments/[\w]+"
```

→ Extracts post links

```python
r"(\d+ comments)"
```

→ Extracts number of comments

5. **Save to Structured Format**

   * Convert matches into dicts
   * Store to `.json` and `.csv`

### Output Fields:

* `title`
* `author`
* `subreddit`
* `score`
* `comments`
* `post_link`

This alternate scraping method is ideal for bulk collection without rate limits or PRAW credentials.

---

## 🧠 Developer Tips

* **Always pre compile regex** with flags (`re.IGNORECASE`, `re.DOTALL`) for performance.
* \*\*Use \*\***`re.findall()`** to extract lists of entities (e.g., hashtags, links).
* \*\*Use ****`re.search()`**** or \*\***`re.match()`** for locating single items like timestamps.

```python
# Sample: Extracting a hashtag from a tweet
re.findall(r"#\w+", tweet_text)
```

* **Pre-check PDF contents** with `.extract_text()` before applying regex.
* **Structure your pipeline** with reusable steps: `extract → filter → transform → save`

---

