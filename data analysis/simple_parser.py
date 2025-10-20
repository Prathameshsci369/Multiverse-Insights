# simple_parser.py
import re
from typing import Dict, Any, List, Optional

import re
from typing import Dict, Any, List, Tuple

def extract_sentiment_final_v3(text: str, max_reasoning_distance: int = 6000, debug: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Robust sentiment extraction that handles variants like:
      - "SENTIMENT_NEGATIVE: 100% - Explanation..."
      - "Negative Sentiment: 100%\nReasoning: ..."
      - "Negative: 100% - short note"
    Returns dict with 'positive','negative','neutral' each having 'percentage' and 'reasoning'.
    """
    def dprint(*a, **k):
        if debug:
            print("[SENT-DBG]", *a, **k)

    # defaults
    out = {
        "positive": {"percentage": 0, "reasoning": "No positive sentiment detected in the text."},
        "negative": {"percentage": 0, "reasoning": "No negative sentiment detected in the text."},
        "neutral":  {"percentage": 0, "reasoning": "No neutral sentiment detected in the text."}
    }
    if not isinstance(text, str) or not text.strip():
        return out

    s = text.replace("\r\n", "\n")

    # 1) Robust header regex:
    # - matches optional "SENTIMENT_" prefix or "Sentiment" suffix
    # - captures inline trailing reason if present after '-' or '—' or ':' on same line
    header_re = re.compile(
        r'(?im)^(?:SENTIMENT[_\s\-]*)?(Positive|Negative|Neutral|POSITIVE|NEGATIVE|NEUTRAL)'
        r'(?:\s+Sentiment)?\s*[:\-]?\s*([\d\.]+)\s*%?(?:\s*[-–—]\s*(.+))?',
        re.MULTILINE
    )

    headers = []
    for m in header_re.finditer(s):
        typ = m.group(1).lower()
        pct_raw = m.group(2)
        inline_reason = m.group(3)
        headers.append((typ, pct_raw, m.start(), m.end(), inline_reason))
    dprint("Found headers (incl. inline reason if any):", headers)

    # 2) Find all explicit 'Reasoning:' blocks and spans
    reasoning_spans: List[Tuple[int,int,str]] = []
    for m in re.finditer(r'(?i)\bReasoning\b\s*[:\-]?\s*', s):
        start = m.end()
        # end at next double newline or next ALLCAPS header or cap length
        candidates = []
        dbl = re.search(r'\n\s*\n', s[start:])
        if dbl:
            candidates.append(start + dbl.start())
        allcaps = re.search(r'\n[A-Z0-9 \-]{3,}(\n|:)', s[start:])
        if allcaps:
            candidates.append(start + allcaps.start())
        end = min(candidates) if candidates else min(len(s), start + 3000)
        snippet = s[start:end].strip()
        reasoning_spans.append((start, end, snippet))
    dprint("Found reasoning spans count:", len(reasoning_spans))

    # helper to choose nearest reasoning block
    def choose_reasoning_for(header_start: int, header_end: int) -> str:
        # prefer reasoning starting after header_end within distance
        after = [(start - header_end, start, snippet) for (start, end, snippet) in reasoning_spans if 0 <= start - header_end <= max_reasoning_distance]
        if after:
            after.sort(key=lambda x: x[0])
            return after[0][2]
        # else prefer nearest before header_start
        before = [(header_start - start, start, snippet) for (start, end, snippet) in reasoning_spans if 0 < header_start - start <= max_reasoning_distance]
        if before:
            before.sort(key=lambda x: x[0])
            return before[0][2]
        return None

    # 3) Assign reasonings (use inline reason first if present)
    if headers:
        for typ, pct_raw, hstart, hend, inline in headers:
            try:
                pct = float(pct_raw)
                pct = int(pct) if float(pct).is_integer() else pct
            except:
                pct = 0
            out[typ]["percentage"] = pct

            if inline and inline.strip():
                # inline reason present on same line after '-'
                reason = inline.strip()
                # normalize whitespace
                clean = re.sub(r'\s*\n\s*', ' ', reason).strip()
                out[typ]["reasoning"] = clean[:1500] + ("..." if len(clean) > 1500 else "")
                dprint(f"Attached INLINE reasoning to {typ}: {out[typ]['reasoning'][:120]}")
                continue

            # else try nearby explicit Reasoning: blocks
            chosen = choose_reasoning_for(hstart, hend)
            if chosen:
                clean = re.sub(r'\s*\n\s*', ' ', chosen).strip()
                out[typ]["reasoning"] = clean[:1500] + ("..." if len(clean) > 1500 else "")
                dprint(f"Attached block reasoning to {typ}: {out[typ]['reasoning'][:120]}")
                continue

            # fallback: capture short snippet after the header (1-2 sentences)
            small = s[hend: hend + 500]
            sents = re.split(r'(?<=[\.\?\!])\s+', small.strip())
            if sents and sents[0].strip():
                fallback = ' '.join(sents[:2]).strip()
                out[typ]["reasoning"] = fallback[:1500] + ("..." if len(fallback) > 1500 else "")
                dprint(f"Used fallback reasoning for {typ}: {out[typ]['reasoning'][:120]}")
            else:
                dprint(f"No reasoning found for {typ}; leaving default.")

    else:
        # no headers found — try coarse extraction
        dprint("No sentiment headers found; attempting coarse extraction")
        coarse = re.findall(r'(?i)(Positive|Negative|Neutral)\s*[:\-]?\s*([\d\.]+)\s*%?', s)
        for typ_raw, pct_raw in coarse:
            typ = typ_raw.lower()
            try:
                pct = float(pct_raw)
                pct = int(pct) if float(pct).is_integer() else pct
            except:
                pct = 0
            out[typ]["percentage"] = pct
        if reasoning_spans:
            longest = max(reasoning_spans, key=lambda t: len(t[2]))[2]
            max_typ = max(out.keys(), key=lambda k: out[k]["percentage"])
            out[max_typ]["reasoning"] = re.sub(r'\s*\n\s*', ' ', longest).strip()[:1500]
            dprint(f"Attached longest reasoning to {max_typ}")

    return out


# --- Add this helper function near other helpers in simple_parser.py ---
import re
from typing import List, Dict, Optional

def _extract_entity_from_phrase(phrase: str) -> Optional[str]:
    """
    Try to identify a probable entity name in a free-text phrase.
    Strategies (in order):
      1) look for 'like <Name>' or 'such as <Name>' patterns (captures the following phrase).
      2) fallback: find the last capitalized multi-word phrase (e.g., 'Aurangzeb', 'Shivaji Maharaj').
    Returns the entity string or None.
    """
    if not phrase or not isinstance(phrase, str):
        return None
    # 1) 'like X' or 'such as X'
    m = re.search(r'(?i)\b(?:like|such as|e\.g\.|eg[:\s])\s+([A-Z][\w\.\-]*(?:\s+[A-Z][\w\.\-]*)*)', phrase)
    if m:
        return m.group(1).strip()

    # 2) find last capitalized sequence (two or more words or single proper noun)
    caps = re.findall(r'([A-Z][\w\.\-]*(?:\s+[A-Z][\w\.\-]*)*)', phrase)
    if caps:
        # choose last item that looks like a name and is not a common word (heuristic)
        cand = caps[-1].strip()
        # simple filter: don't return short words like "The" (though pattern avoids that)
        if len(cand) > 1:
            return cand
    return None

def parse_relationships_section(text: str, debug_fn=None) -> List[Dict[str, Optional[str]]]:
    """
    Parse the RELATIONSHIPS section and return list of {entity1, relationship, entity2}.
    - debug_fn is an optional callable(msg) for debug logging (e.g., self.debug_print).
    """
    if debug_fn is None:
        debug_fn = lambda *_: None

    rels: List[Dict[str, Optional[str]]] = []

    # 1) find the RELATIONSHIPS: section (up to next ALL-CAPS header or end)
    m = re.search(r'(?mi)^\s*RELATIONSHIPS\s*:\s*(.+?)(?=\n^[A-Z0-9 _\-]{3,}?:|\n{2,}|\Z)', text, re.DOTALL | re.MULTILINE)
    if not m:
        # fallback: look for a line starting with "RELATIONSHIPS:" inline
        m = re.search(r'(?mi)RELATIONSHIPS\s*:\s*(.+)', text)
        if not m:
            debug_fn("No RELATIONSHIPS section found")
            return rels

    body = m.group(1).strip()
    debug_fn(f"Raw relationships body: {body[:200]}")

    # 2) split into parts by semicolon, newline, or bullet
    parts = [p.strip() for p in re.split(r';|\n|•', body) if p.strip()]
    debug_fn(f"Relationship parts: {parts}")

    for part in parts:
        # Try A -> B or A -> description
        p = part.strip()
        # common arrow variants
        arrow_match = re.match(r'^\s*(.+?)\s*(?:->|→)\s*(.+)$', p)
        if arrow_match:
            left = arrow_match.group(1).strip()
            right = arrow_match.group(2).strip()
            # if right is short and looks like an entity, treat it as entity2
            # else, attempt to extract entity2 from the right phrase
            entity2 = None
            # if right has comma or 'like' or 'such as' try to extract embedded entity
            entity2 = _extract_entity_from_phrase(right)
            relationship_text = right
            rels.append({"entity1": left, "relationship": relationship_text, "entity2": entity2})
            debug_fn(f"Parsed arrow: {left} -> {relationship_text} (entity2={entity2})")
            continue

        # Try A : B  or A - B  or A — B
        colon_match = re.match(r'^\s*(.+?)\s*(?:[:\-–—])\s*(.+)$', p)
        if colon_match:
            left = colon_match.group(1).strip()
            right = colon_match.group(2).strip()
            entity2 = _extract_entity_from_phrase(right)
            relationship_text = right
            rels.append({"entity1": left, "relationship": relationship_text, "entity2": entity2})
            debug_fn(f"Parsed colon/dash: {left} -> {relationship_text} (entity2={entity2})")
            continue

        # Try pattern "A - verb phrase about B" (no explicit separator)
        # As a last resort, attempt to split at the first verb-ish token ("is", "was", "accused", etc.)
        verb_split = re.split(r'\b(is|was|are|accused|accuses|accusing|alleged|attacks|criticizes|supports|oppose|called)\b', p, maxsplit=1, flags=re.IGNORECASE)
        if len(verb_split) >= 3:
            left = verb_split[0].strip()
            verb = verb_split[1].strip()
            rest = verb_split[2].strip()
            entity2 = _extract_entity_from_phrase(rest)
            rels.append({"entity1": left, "relationship": f"{verb} {rest}".strip(), "entity2": entity2})
            debug_fn(f"Parsed verb-split: {left} -> {verb} {rest} (entity2={entity2})")
            continue

        # fallback: store full part as relationship text
        rels.append({"entity1": None, "relationship": p, "entity2": None})
        debug_fn(f"Fallback relationship parse: {p}")

    return rels




class SimpleFormatParser:
    """
    A parser for the simple, structured output format.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def debug_print(self, message: str):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def parse_simple_format(self, text: str) -> Dict[str, Any]:
        """
        Improved parser for the simple structured output.
        - Robustly extracts executive summary
        - Extracts full sentiment block (positive/negative/neutral + reasonings)
        - Extracts topics while avoiding overlap with entities/relationships
        - Extracts entities, relationships, anomalies, controversy score
        """
        self.debug_print("Starting to parse simple format (improved)")

        result = {"multiverse_combined": {}}

        # --- Executive summary ---
        # exec_match = re.search(r'\[EXECUTIVE_SUMMARY\]\s*(.+?)(?=\n[A-Z0-9_ \-]{3,}?:|\n$)', text, re.DOTALL)
        # if exec_match:
        #     result["multiverse_combined"]["executive_summary"] = exec_match.group(1).strip()
        #     self.debug_print("Extracted executive summary")
            
        # Try to capture between tags or until SENTIMENT_/TOPICS_/ENTITIES_
        executive_summary_match = re.search(
            r'<EXECUTIVE_SUMMARY>\s*(.*?)(?=SENTIMENT_|TOPICS:|ENTITIES:|$)', 
            text, 
            re.DOTALL
        )

        if executive_summary_match:
            result["multiverse_combined"]["executive_summary"] = executive_summary_match.group(1).strip()
        else:
            # Fallback for LLM outputs without tags
            executive_summary_match = re.search(
                r'EXECUTIVE_SUMMARY\s*[:\n]?\s*(.*?)(?=SENTIMENT_|$)',
                text,
                re.DOTALL
            )
            if executive_summary_match:
                result["multiverse_combined"]["executive_summary"] = executive_summary_match.group(1).strip()


        # --- Sentiment analysis (improved) ---
        # sentiment_struct = extract_sentiment_block_v2(text)
        # result["multiverse_combined"]["sentiment_analysis"] = sentiment_struct
        # self.debug_print("Extracted sentiment (improved)")

        sentiment_struct = extract_sentiment_final_v3(text, debug=True)
        result["multiverse_combined"]["sentiment_analysis"] = sentiment_struct


        # --- Topics ---
        # Extract the TOPICS: line/section until next header like ENTITIES: or a blank line or the next all-caps header
        topics_match = re.search(r'(?m)^TOPICS:\s*(.+?)(?=\n^[A-Z0-9 _\-]{3,}?:|\nKEY TOPICS|\n$)', text, re.DOTALL)
        topics_list: List[str] = []
        if topics_match:
            topics_str = topics_match.group(1).strip()
            # sometimes topics are comma-separated or bullet-listed; normalize
            # split by commas or line-break bullets
            if '\n' in topics_str and '•' in topics_str:
                # bullet list found — split on bullets or newlines
                items = [t.strip(" •\t") for t in re.split(r'[\n•]+', topics_str) if t.strip()]
            else:
                items = [t.strip() for t in re.split(r',|\n', topics_str) if t.strip()]
            topics_list = items
            self.debug_print("Extracted topics (raw)")

        # --- Entities ---
        entities_match = re.search(r'(?m)^ENTITIES:\s*(.+?)(?=\n^[A-Z0-9 _\-]{3,}?:|\n$)', text, re.DOTALL)
        entities_list: List[str] = []
        if entities_match:
            entities_str = entities_match.group(1).strip()
            entities_list = [e.strip() for e in re.split(r',|\n|•', entities_str) if e.strip()]
            self.debug_print("Extracted entities")

        
        # Extract relationships - use a more specific pattern to avoid overlap
        relationships_match = re.search(r'RELATIONSHIPS:\s*(.+?)(?=\nANOMALIES:)', text, re.DOTALL)
        if relationships_match:
            relationships_str = relationships_match.group(1).strip()
            relationships = []
            
            # Split by semicolons
            rel_parts = [rel.strip() for rel in relationships_str.split(';') if rel.strip()]
            
            for rel_part in rel_parts:
                # Match pattern: entity1 -> description
                rel_match = re.match(r'(.+?)\s*->\s*(.+)', rel_part)
                if rel_match:
                    entity = rel_match.group(1).strip()
                    description = rel_match.group(2).strip()
                    relationships.append({
                        "entity": entity,
                        "description": description
                    })
            
            result["multiverse_combined"]["relationship_extraction"] = relationships
            self.debug_print("Extracted relationships")
        


        # --- Anomalies ---
        anomalies_match = re.search(r'(?m)^ANOMALIES:\s*(.+?)(?=\n^[A-Z0-9 _\-]{3,}?:|\n$)', text, re.DOTALL)
        anomalies = []
        if anomalies_match:
            anomalies_str = anomalies_match.group(1).strip()
            if anomalies_str.lower() not in ("none", "no anomalies detected"):
                # split by semicolon or newline bullets
                anomalies = [{"description": a.strip()} for a in re.split(r';|\n|•', anomalies_str) if a.strip()]
            self.debug_print("Extracted anomalies")

        # --- Controversy score ---
        controversy = {}
        controversy_match = re.search(r'(?m)^CONTROVERSY_SCORE:\s*([\d.]+)\s*-\s*(.+?)(?=\n|$)', text, re.DOTALL)
        if controversy_match:
            try:
                score = float(controversy_match.group(1))
                controversy = {"value": score, "explanation": controversy_match.group(2).strip()}
                self.debug_print("Extracted controversy score")
            except ValueError:
                self.debug_print(f"Failed to parse controversy score: {controversy_match.group(1)}")

        # --- Clean topics: remove items that are actually entities or relationship fragments ---
        cleaned_topics = []
        entities_lower = {e.lower() for e in entities_list}
        rel_fragments = set()
        for r in relationships:
            if r.get("entity1"):
                rel_fragments.add(r["entity1"].lower())
            if r.get("entity2"):
                rel_fragments.add(str(r["entity2"]).lower())
            if r.get("relationship"):
                # split relationship text into tokens and add those tokens that are multi-word phrases
                rel_fragments.update([token.strip().lower() for token in re.split(r'[,:;()-]', r["relationship"]) if token.strip()])

        for t in topics_list:
            tl = t.lower()
            # skip if the topic exactly matches an entity or is a short fragment from relationships
            if tl in entities_lower:
                self.debug_print(f"Skipping topic '{t}' because it matches an entity")
                continue
            # skip if topic is contained in any relationship fragment (avoid redundancy)
            skip = False
            for frag in rel_fragments:
                if frag and frag in tl:
                    self.debug_print(f"Skipping topic '{t}' because it overlaps relationship fragment '{frag}'")
                    skip = True
                    break
            if not skip:
                cleaned_topics.append(t)

        # fallbacks to ensure keys exist
        result["multiverse_combined"]["topics"] = cleaned_topics
        result["multiverse_combined"]["entity_recognition"] = entities_list
        result["multiverse_combined"]["relationship_extraction"] = relationships
        result["multiverse_combined"]["anomaly_detection"] = anomalies
        if controversy:
            result["multiverse_combined"]["controversy_score"] = controversy
        else:
            result["multiverse_combined"]["controversy_score"] = {"value": 0.0, "explanation": ""}

        # ensure sentiment exists
        if "sentiment_analysis" not in result["multiverse_combined"]:
            result["multiverse_combined"]["sentiment_analysis"] = sentiment_struct

        # raw_text fallback snippet
        if "raw_text" not in result["multiverse_combined"]:
            # capture a short snippet from top of file
            result["multiverse_combined"]["raw_text_snippet"] = text.strip()[:500]

        return result

    
    def parse_and_display(self, file_path: str):
        """
        Parse a file with the simple format and display the results.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the content
            parsed_data = self.parse_simple_format(content)
            
            # Display the results
            self.display_results(parsed_data)
            
            # Save as JSON for reference
            import json
            with open("parsed_output.json", 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=4, ensure_ascii=False)
            
            print("\n✅ Parsed data saved to parsed_output.json")
            
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except Exception as e:
            print(f"Error: {e}")
    
    def display_results(self, data: Dict[str, Any]):
        """
        Display the parsed results in a human-readable format.
        """
        if "multiverse_combined" not in data:
            print("No valid data found.")
            return
        
        analysis_data = data["multiverse_combined"]
        
        print("\n" + "="*80)
        print("ANALYSIS RESULTS".center(80))
        print("="*80)
        
        # Executive Summary
        if "executive_summary" in analysis_data:
            print("\nEXECUTIVE SUMMARY")
            print("-" * 80)
            print(analysis_data["executive_summary"])
        
        # Sentiment Analysis
        if "sentiment_analysis" in analysis_data:
            print("\nSENTIMENT ANALYSIS")
            print("-" * 80)
            sentiment = analysis_data["sentiment_analysis"]
            for sentiment_type, details in sentiment.items():
                percentage = details.get("percentage", 0)
                reasoning = details.get("reasoning", "")
                print(f"\n{sentiment_type.capitalize()} Sentiment: {percentage}%")
                print(f"Reasoning: {reasoning}")
        
        # Topics
        if "topics" in analysis_data:
            print("\nKEY TOPICS")
            print("-" * 80)
            for topic in analysis_data["topics"]:
                print(f"• {topic}")
        
        # Entity Recognition
        if "entity_recognition" in analysis_data:
            print("\nRECOGNIZED ENTITIES")
            print("-" * 80)
            for entity in analysis_data["entity_recognition"]:
                print(f"• {entity}")
        
        # Relationship Extraction
        if "relationship_extraction" in analysis_data:
            print("\nRELATIONSHIPS")
            print("-" * 80)
            for rel in analysis_data["relationship_extraction"]:
                if "entity" in rel and "description" in rel:
                    print(f"• {rel['entity']}: {rel['description']}")
        
        # Anomaly Detection
        if "anomaly_detection" in analysis_data:
            print("\nDETECTED ANOMALIES")
            print("-" * 80)
            if analysis_data["anomaly_detection"]:
                for anomaly in analysis_data["anomaly_detection"]:
                    if "description" in anomaly:
                        print(f"• {anomaly['description']}")
            else:
                print("• No anomalies detected")
        
        # Controversy Score
        if "controversy_score" in analysis_data:
            print("\nCONTROVERSY SCORE")
            print("-" * 80)
            controversy = analysis_data["controversy_score"]
            value = controversy.get("value", 0)
            explanation = controversy.get("explanation", "")
            print(f"Score: {value}/1.0")
            print(f"Explanation: {explanation}")
        
        print("\n" + "="*80)
        print("END OF ANALYSIS".center(80))
        print("="*80)

# Create a global instance of the parser
parser = SimpleFormatParser()

# Convenience functions
def parse_and_display(file_path: str, debug: bool = False):
    """
    Parse a file with the simple format and display the results.
    """
    if debug:
        parser.debug = True
    parser.parse_and_display(file_path)

def parse_simple_format(text: str, debug: bool = False) -> Dict[str, Any]:
    """
    Parse text in the simple format.
    """
    if debug:
        parser.debug = True
    return parser.parse_simple_format(text)

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        debug_mode = "--debug" in sys.argv
        parse_and_display(file_path, debug=debug_mode)
    else:
        print("Usage: python simple_parser.py <file_path> [--debug]")