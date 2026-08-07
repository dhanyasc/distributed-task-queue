"""
Task processors – pluggable processing logic for different task types.

Each processor implements `process(payload) -> dict` and handles one task type.
"""

import time
import math
import re
from typing import Dict
from collections import Counter


class BaseProcessor:
    """Base class for task processors."""

    def process(self, payload: dict) -> dict:
        raise NotImplementedError


class MLInferenceProcessor(BaseProcessor):
    """
    Simulates ML model inference.
    In production, replace with actual model loading and prediction.
    Payload: {"text": "...", "model": "sentiment|classification|ner"}
    """

    def process(self, payload: dict) -> dict:
        text = payload.get("text", "")
        model = payload.get("model", "sentiment")

        if model == "sentiment":
            return self._sentiment(text)
        elif model == "classification":
            return self._classify(text)
        elif model == "ner":
            return self._ner(text)
        else:
            raise ValueError(f"Unknown model: {model}")

    def _sentiment(self, text: str) -> dict:
        # Simple rule-based sentiment (replace with real model)
        positive = {"good", "great", "excellent", "amazing", "love", "best", "happy", "wonderful"}
        negative = {"bad", "terrible", "awful", "hate", "worst", "sad", "horrible", "poor"}
        words = set(text.lower().split())

        pos_count = len(words & positive)
        neg_count = len(words & negative)
        total = pos_count + neg_count or 1

        if pos_count > neg_count:
            label, score = "positive", pos_count / total
        elif neg_count > pos_count:
            label, score = "negative", neg_count / total
        else:
            label, score = "neutral", 0.5

        time.sleep(0.1)  # Simulate inference time
        return {"label": label, "score": round(score, 3), "model": "sentiment-v1"}

    def _classify(self, text: str) -> dict:
        categories = {
            "technology": {"ai", "machine", "learning", "software", "data", "cloud", "api", "code"},
            "finance": {"stock", "market", "trading", "investment", "bank", "revenue", "profit"},
            "science": {"research", "experiment", "hypothesis", "study", "analysis", "theory"},
            "health": {"medical", "health", "disease", "treatment", "patient", "clinical"},
        }
        words = set(text.lower().split())
        scores = {cat: len(words & kws) for cat, kws in categories.items()}
        best = max(scores, key=scores.get) if any(scores.values()) else "general"
        time.sleep(0.05)
        return {"category": best, "confidence": min(scores.get(best, 0) / 3, 1.0)}

    def _ner(self, text: str) -> dict:
        # Simple pattern-based NER
        entities = []
        # Capitalized words as potential entities
        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            entities.append({"text": match.group(), "type": "ENTITY", "start": match.start()})
        time.sleep(0.05)
        return {"entities": entities[:20], "count": len(entities)}


class DataProcessingProcessor(BaseProcessor):
    """
    Data transformation and analysis.
    Payload: {"data": [...], "operation": "aggregate|transform|filter"}
    """

    def process(self, payload: dict) -> dict:
        data = payload.get("data", [])
        operation = payload.get("operation", "aggregate")

        if operation == "aggregate":
            return self._aggregate(data)
        elif operation == "transform":
            return self._transform(data, payload.get("transform_fn", "uppercase"))
        elif operation == "filter":
            return self._filter(data, payload.get("condition", {}))
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _aggregate(self, data: list) -> dict:
        if not data:
            return {"count": 0, "summary": {}}

        if all(isinstance(d, (int, float)) for d in data):
            return {
                "count": len(data),
                "sum": sum(data),
                "mean": sum(data) / len(data),
                "min": min(data),
                "max": max(data),
                "std_dev": round(math.sqrt(sum((x - sum(data)/len(data))**2 for x in data) / len(data)), 4),
            }
        return {"count": len(data), "types": dict(Counter(type(d).__name__ for d in data))}

    def _transform(self, data: list, fn: str) -> dict:
        if fn == "uppercase":
            result = [str(d).upper() for d in data]
        elif fn == "lowercase":
            result = [str(d).lower() for d in data]
        elif fn == "double":
            result = [d * 2 if isinstance(d, (int, float)) else d for d in data]
        else:
            result = data
        return {"original_count": len(data), "transformed": result}

    def _filter(self, data: list, condition: dict) -> dict:
        field = condition.get("field")
        op = condition.get("op", "eq")
        value = condition.get("value")

        if not field or value is None:
            return {"filtered": data, "count": len(data)}

        filtered = []
        for item in data:
            if isinstance(item, dict) and field in item:
                v = item[field]
                if op == "eq" and v == value:
                    filtered.append(item)
                elif op == "gt" and v > value:
                    filtered.append(item)
                elif op == "lt" and v < value:
                    filtered.append(item)
                elif op == "contains" and isinstance(v, str) and value in v:
                    filtered.append(item)

        return {"filtered": filtered, "count": len(filtered), "original_count": len(data)}


class TextAnalysisProcessor(BaseProcessor):
    """
    Text analytics: word frequency, readability, summarization.
    Payload: {"text": "...", "analysis": "frequency|readability|summary"}
    """

    def process(self, payload: dict) -> dict:
        text = payload.get("text", "")
        analysis = payload.get("analysis", "frequency")

        if analysis == "frequency":
            return self._frequency(text)
        elif analysis == "readability":
            return self._readability(text)
        elif analysis == "summary":
            return self._summary(text)
        else:
            raise ValueError(f"Unknown analysis: {analysis}")

    def _frequency(self, text: str) -> dict:
        words = re.findall(r'\b\w+\b', text.lower())
        stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or"}
        filtered = [w for w in words if w not in stop and len(w) > 2]
        counts = Counter(filtered).most_common(20)
        return {"total_words": len(words), "unique_words": len(set(words)), "top_words": dict(counts)}

    def _readability(self, text: str) -> dict:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        syllables = sum(self._count_syllables(w) for w in words)

        if not sentences or not words:
            return {"flesch_score": 0, "grade_level": "N/A"}

        avg_sentence_len = len(words) / len(sentences)
        avg_syllables = syllables / len(words)

        flesch = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables
        return {
            "flesch_score": round(max(0, min(100, flesch)), 1),
            "sentence_count": len(sentences),
            "word_count": len(words),
            "avg_sentence_length": round(avg_sentence_len, 1),
        }

    def _summary(self, text: str) -> dict:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        # Take first and most "important" sentences (by length as proxy)
        if len(sentences) <= 3:
            summary = " ".join(sentences)
        else:
            ranked = sorted(sentences, key=len, reverse=True)
            summary = " ".join(ranked[:3])
        return {"summary": summary, "original_sentences": len(sentences), "summary_sentences": min(3, len(sentences))}

    def _count_syllables(self, word: str) -> int:
        word = word.lower().rstrip("e")
        return max(1, len(re.findall(r'[aeiouy]+', word)))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PROCESSORS = {
    "ml_inference": MLInferenceProcessor(),
    "data_processing": DataProcessingProcessor(),
    "text_analysis": TextAnalysisProcessor(),
}


def get_processor(task_type: str) -> BaseProcessor:
    proc = _PROCESSORS.get(task_type)
    if proc is None:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(_PROCESSORS.keys())}")
    return proc
