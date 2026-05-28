import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class Route(Enum):
    DIRECT = "direct"
    DECOMPOSE = "decompose"
    HYDE = "hyde"

# Initialize spaCy globally so we only pay the 15MB load penalty once
try:
    import spacy
    # Disable pipelines we don't need to keep execution time under 15ms
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
except ImportError:
    nlp = None
    logger.warning("spaCy is not installed. Falling back to naive string matching.")
except OSError:
    nlp = None
    logger.warning("spaCy model 'en_core_web_sm' is not downloaded. Falling back to naive string matching.")

def route_query(query: str) -> Route:
    """
    Sub-15ms router to classify the query path using NLP dependency parsing.
    """
    query_lower = query.lower().strip()
    words = query_lower.split()
    
    # Heuristic 1: Sparse/Short queries (<= 4 words)
    # Short queries lack semantic context, making them perfect candidates for HyDE expansion
    if len(words) <= 4:
        return Route.HYDE
        
    # Heuristic 2: Compositional / Compare queries
    if nlp is not None:
        doc = nlp(query_lower)
        for token in doc:
            # Rule 1: Explicit comparison roots
            if token.text in ["vs", "versus", "compare", "between"]:
                return Route.DECOMPOSE
                
            # Rule 2: Syntactic conjunctions of Nouns/Proper Nouns
            if token.dep_ == "conj" and token.head.pos_ in ["NOUN", "PROPN"]:
                return Route.DECOMPOSE
    else:
        # Fallback if spaCy failed to load
        decomposition_keywords = [
            " vs ", " versus ", " compare ", " compared ", " differences ", " both "
        ]
        if any(keyword in query_lower for keyword in decomposition_keywords):
            return Route.DECOMPOSE
        
    # Heuristic 3: Strict proximity regex for metadata extraction (Look for years)
    metadata_pattern = re.compile(r"(?:published|available|released|from|since|before|after|in)\s+(?:year\s+)?(19\d{2}|20\d{2})")
    if metadata_pattern.search(query_lower):
        return Route.DECOMPOSE
        
    # Default fallback: Long, highly-descriptive queries already have enough semantic density
    return Route.DIRECT
