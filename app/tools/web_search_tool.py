import re
import time
import json
import spacy
import logging
from typing import List, Dict
from langchain.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain_community.tools import DuckDuckGoSearchResults

# Module-level logger
logger = logging.getLogger(__name__)


class SearchTool:
    """Search tool with query classification and optimization."""

    def __init__(self, max_results: int = 5):
        self.search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=10)
        self.base_tool = DuckDuckGoSearchResults(api_wrapper=self.search_wrapper, output_format="json")

        self.wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=3000,
            load_all_available_meta=True,
        )
        self.wiki_tool = WikipediaQueryRun(api_wrapper=self.wiki_wrapper)

        self.nlp = spacy.load("en_core_web_sm")  # Lightweight NLP model
        self.max_results = max_results
        self.query_cache: Dict[str, List[Dict]] = {}

        self.query_types = {
            "definitional": ["what is", "define", "definition", "meaning"],
            "statistical": ["how many", "number", "count", "statistics", "data"],
            "comparative": ["vs", "versus", "compare", "comparison"],
            "causal": ["why", "reason", "cause", "because"],
        }

        logger.info("SearchTool initialized")

    def _classify_query(self, question: str) -> str:
        """Classify query type based on keywords."""
        logger.info(f"Classifying query: {question}")
        q_lower = question.lower()
        for q_type, keywords in self.query_types.items():
            if any(k in q_lower for k in keywords):
                return q_type
        return "general"

    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms using spaCy."""
        logger.debug("Extracting key terms from query")
        doc = self.nlp(question)
        key_terms = [
            token.text
            for token in doc
            if token.pos_ in ["NOUN", "PROPN", "NUM"] and not token.is_stop
        ]
        return list(set(key_terms))  # Remove duplicates

    def _optimize_query(self, question: str, query_type: str) -> List[str]:
        """Generate optimized search queries based on type."""
        logger.debug("Optimizing query")
        key_terms = self._extract_key_terms(question)
        queries = [re.sub(r"[^\w\s]", " ", question).strip()]  # Cleaned original

        if key_terms:
            queries.append(" ".join(key_terms[:5]))  # Key terms only

        if query_type == "statistical":
            queries.append(" ".join(key_terms[:3]) + " statistics data count")
        elif query_type == "definitional":
            queries.append(" ".join(key_terms[:2]) + " definition meaning")
        elif query_type == "comparative":
            queries.append(" ".join(key_terms[:3]) + " comparison review")
        elif query_type == "causal":
            queries.append(" ".join(key_terms[:3]) + " reason cause explanation")

        return list(set(queries))

    def _rank_results(self, results: List[Dict], key_terms: List[str]) -> List[Dict]:
        """Rank results by keyword overlap and source quality."""
        if not results:
            return []
        seen_snippets = set()
        scored = []
        key_terms_lower = [t.lower() for t in key_terms]
        for r in results:
            snippet = (r.get("snippet", "") + " " + r.get("title", "")).lower()
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            score = sum(snippet.count(term) for term in key_terms_lower)
            if len(snippet) < 30:
                score -= 1
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for score, r in scored]

    def run(self, question: str) -> str:
        """Main pipeline: classify → optimize → search → rank."""
        logger.info("Running SearchTool pipeline")

        if question in self.query_cache:
            logger.debug("Returning cached results")
            return json.dumps(self.query_cache[question], indent=2)

        query_type = self._classify_query(question)

        # if query_type == "definitional":
        #     try:
        #         wiki_result = self.wiki_tool.run(question)
        #         results = [{"title": "Wikipedia", "snippet": wiki_result, "link": ""}]
        #         self.query_cache[question] = results
        #         return json.dumps(results, indent=2)
        #     except Exception:
        #         logger.exception("Wikipedia search failed, falling back to web search")

        queries = self._optimize_query(question, query_type)
        key_terms = self._extract_key_terms(question)
        all_results = []

        for q in queries[:3]:
            logger.info(f"Searching for: {q}")
            try:
                results = self.base_tool.invoke(q)
                logger.debug(f"Raw results: {results}")
                if isinstance(results, str):
                    try:
                        results = json.loads(results)
                    except json.JSONDecodeError:
                        logger.exception("Failed to parse results for query")
                        results = [
                            {
                                "title": "Search Result",
                                "snippet": results,
                                "link": "",
                                "Searching For": q,
                            }
                        ]
                all_results.extend(results if isinstance(results, list) else [])
                time.sleep(0.5)  # Avoid API rate limits
            except Exception:
                logger.exception("Web search failed")

        if not all_results:
            logger.warning("No search results found")
            result = [{"title": "", "snippet": "No search results found.", "link": ""}]
            self.query_cache[question] = result
            return json.dumps(result, indent=2)

        ranked = self._rank_results(all_results, key_terms)
        top_results = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", ""),
            }
            for r in ranked[: self.max_results]
        ]
        self.query_cache[question] = top_results
        return json.dumps(top_results, indent=2)
