import os
import joblib
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

class MLQueryRouter:
    def __init__(self, model_path: str = "data/router_dataset/query_router_model.joblib"):
        self.classifier = None
        if os.path.exists(model_path):
            try:
                self.classifier = joblib.load(model_path)
                logger.info(f"Successfully loaded ML router from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ML router: {e}")
        else:
            logger.warning(f"ML router model not found at {model_path}. Falling back to heuristics.")

    def route(self, query: str, query_vector: list = None) -> Route:
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Heuristic 1: Sparse/Short queries (<= 4 words)
        if len(words) <= 4:
            logger.info(f"Query length ({len(words)}) <= 4 words, defaulting to Route.HYDE")
            return Route.HYDE
            
        # ML Routing (if loaded and vector provided)
        if self.classifier is not None and query_vector is not None:
            pred = self.classifier.predict([query_vector])[0]
            if pred == 1:
                logger.info(f"ML Router predicted class {pred} -> Route.DECOMPOSE")
                return Route.DECOMPOSE
            else:
                logger.info(f"ML Router predicted class {pred} -> Route.DIRECT")
                return Route.DIRECT
                
        # Heuristic 2: Compositional / Compare queries
        if nlp is not None:
            doc = nlp(query_lower)
            for token in doc:
                # Rule 1: Explicit comparison roots
                if token.text in ["vs", "versus", "compare", "between"]:
                    logger.info(f"spaCy heuristic matched on '{token.text}', returning Route.DECOMPOSE")
                    return Route.DECOMPOSE
                    
                # Rule 2: Syntactic conjunctions of Nouns/Proper Nouns
                if token.dep_ == "conj" and token.head.pos_ in ["NOUN", "PROPN"]:
                    logger.info("spaCy heuristic matched conjunction, returning Route.DECOMPOSE")
                    return Route.DECOMPOSE
        else:
            # Fallback if spaCy failed to load
            decomposition_keywords = [
                " vs ", " versus ", " compare ", " compared ", " differences ", " both "
            ]
            if any(keyword in query_lower for keyword in decomposition_keywords):
                logger.info("Fallback keyword matched, returning Route.DECOMPOSE")
                return Route.DECOMPOSE
            
        # Heuristic 3: Strict proximity regex for metadata extraction (Look for years)
        metadata_pattern = re.compile(r"(?:published|available|released|from|since|before|after|in)\s+(?:year\s+)?(19\d{2}|20\d{2})")
        if metadata_pattern.search(query_lower):
            logger.info("Metadata pattern matched, returning Route.DECOMPOSE")
            return Route.DECOMPOSE
            
        # Default fallback: Long, highly-descriptive queries already have enough semantic density
        logger.info("Default fallback reached, returning Route.DIRECT")
        return Route.DIRECT

def route_query(query: str) -> Route:
    """
    Sub-15ms router to classify the query path using NLP dependency parsing.
    Retained for backward compatibility.
    """
    router = MLQueryRouter()
    return router.route(query)
