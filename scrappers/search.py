# ✅ 2. Import libraries
import re

# ✅ 3. Initialize model
print("⏳ Loading DistilBERT...")
from keybert import KeyBERT
kw_model = KeyBERT('distilbert-base-nli-mean-tokens')
print("✅ Model ready.")

# ✅ 4. Utility: clean and normalize keyphrases
def clean_keyphrases(phrases, min_words=2):
    seen = set()
    results = []
    for phrase, _ in phrases:
        # Lowercase and remove filler/stop words
        phrase = phrase.lower()
        phrase = re.sub(r'\b(i|want|know|need|info|information|about|tell me|something|like|give|looking for)\b', '', phrase)
        phrase = re.sub(r'[^a-zA-Z0-9\s]', '', phrase)  # remove special chars
        phrase = re.sub(r'\s+', ' ', phrase).strip()

        if len(phrase.split()) < min_words:
            continue
        if phrase not in seen:
            results.append(phrase)
            seen.add(phrase)
    return results

# ✅ 5. Main function: generate search-style query
def generate_search_query(text, max_phrases=5):
    # Extract keywords
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(2, 4),
        stop_words='english',
        use_mmr=True,
        diversity=0.7,
        top_n=20
    )


    # Clean results
    clean_phrases = clean_keyphrases(keywords)

    # Return top N joined with commas
    return ", ".join(clean_phrases[:max_phrases])

# ✅ 6. Example usage
texts = [
    "I want the information about NASA and ISRO future space missions, latest developments in AI, and opinions on new space missions.",
    "Climate change is affecting weather patterns globally. Governments are investing in green energy. The economic cost of inaction is high.",
    "Tell me something about ancient civilizations, their agriculture, social hierarchy, and technological advancements."
]

for i, t in enumerate(texts):
    print(f"\n🔍 Search Query {i+1}:")
    print(generate_search_query(t))

class SearchQueryGenerator:
    def __init__(self, model_name='distilbert-base-nli-mean-tokens'):
        """
        Initialize the KeyBERT model for keyword extraction.
        """
        from keybert import KeyBERT
        print("⏳ Loading DistilBERT...")
        self.kw_model = KeyBERT(model_name)
        print("✅ Model ready.")

    def clean_keyphrases(self, phrases, min_words=2):
        """
        Clean and normalize keyphrases by removing filler words and duplicates.
        """
        import re
        seen = set()
        results = []
        for phrase, _ in phrases:
            phrase = phrase.lower()
            phrase = re.sub(r'\b(i|want|know|need|info|information|about|tell me|something|like|give|looking for)\b', '', phrase)
            phrase = re.sub(r'[^a-zA-Z0-9\s]', '', phrase)  # remove special chars
            phrase = re.sub(r'\s+', ' ', phrase).strip()

            if len(phrase.split()) < min_words:
                continue
            if phrase not in seen:
                results.append(phrase)
                seen.add(phrase)
        return results

    def generate_search_query(self, text, max_phrases=5):
        """
        Generate a search-style query from the input text.
        """
        # Extract keywords
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(2, 4),
            stop_words='english',
            use_mmr=True,
            diversity=0.7,
            top_n=20
        )

        # Clean results
        clean_phrases = self.clean_keyphrases(keywords)

        # Return top N joined with commas
        return ", ".join(clean_phrases[:max_phrases])
