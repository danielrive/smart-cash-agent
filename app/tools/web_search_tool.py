import json
import spacy
import logging
from typing import List, Dict

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import UnstructuredURLLoader

# Module-level logger
logger = logging.getLogger(__name__)


class SearchTool:
    """Wikipedia search tool optimized for GAIA benchmark questions."""
    def __init__(self, max_results: int = 10, fetch_top_n: int = 3, max_chars: int = 5000):
        logger.info("Initializing SearchTool (Wikipedia-only)...")
        
        self.wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=3000,
            load_all_available_meta=True,
        )

        # Load spacy model with error handling
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("Spacy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        # Config
        self.max_results = max_results
        self.fetch_top_n = fetch_top_n
        self.max_chars = max_chars

        logger.info("SearchTool initialized successfully.")

    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms using spaCy."""
        logger.debug("Extracting key terms from question")
        doc = self.nlp(question)
        key_terms = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ["NOUN", "PROPN", "NUM"] and not token.is_stop
        ]
        key_terms = list(set(key_terms)) or question.lower().split()
        logger.debug(f"Key terms extracted: {key_terms}")
        return key_terms

    def _rank_results(self, results: List[Dict], key_terms: List[str]) -> List[Dict]:
        """Rank results by keyword overlap and source quality."""
        logger.debug("Ranking results...")
        logger.debug(f"Input: {len(results)} results, {len(key_terms)} key terms")

        if not results:
            return []
        
        seen_snippets = set()
        scored = []
        key_terms_lower = [t.lower() for t in key_terms]
        threshold = max(1, len(key_terms_lower) // 2)
        logger.debug(f"Ranking threshold: {threshold}")

        for r in results:
            snippet = (r.get("snippet", "") + " " + r.get("title", "")).lower()
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            
            # Score by keyword overlap
            score = sum(snippet.count(term) for term in key_terms_lower)
            
            # Penalize very short snippets
            if len(snippet) < 20:
                score -= 1

            if score >= threshold:
                scored.append((score, r))
                logger.debug(f"Result scored {score}: {r.get('title', 'No title')[:50]}")

        scored.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Ranked {len(scored)} results (from {len(results)} total)")
        if scored:
            logger.debug(f"Top result: {scored[0][1].get('title', 'No title')}")
        return [r for score, r in scored]

    def _fetch_full_content(self, urls: List[str]) -> Dict[str, str]:
        """Fetch page content for given URLs."""
        logger.info("Fetching full content")
        if not urls:
            return {}

        try:
            loader = UnstructuredURLLoader(
                urls=urls,
                headers={
                    "User-Agent": "smartcashagent/1.0 (test@smartcash.com; For GAIA benchmark eval via LangGraph)" ## Header is needed to avoid that site blocks trating us as a robot
                },
            )
            docs = loader.load()
        except Exception as e:
            logger.exception("Failed to fetch full content")
            return {}

        url_to_content = {}
        for doc in docs:
            content = doc.page_content.strip()
            if len(content) > self.max_chars:
                content = content[: self.max_chars] + "..."
            url_to_content[doc.metadata.get("source")] = content
        return url_to_content

    def _search_wikipedia(self, question: str) -> List[Dict]:
        """Search Wikipedia with entity extraction for better results."""
        spacy_doc = self.nlp(question)

        # Extract named entities (PERSON, ORG, GPE, WORK_OF_ART, EVENT, etc.)
        entities = [ent.text for ent in spacy_doc.ents if ent.label_ in [
            "PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT"
        ]]
        logger.debug(f"Named entities extracted: {entities}")
        
        # Use entities as search terms if found, otherwise use full question
        search_query = " ".join(entities) if entities else question
        logger.info(f"Wikipedia search query: {search_query}")
        
        try:
            docs = self.wiki_wrapper.load(search_query)
            logger.info(f"Wikipedia returned {len(docs)} documents")
            
            results = []
            for d in docs:
                content = d.page_content[:500]  # trim to avoid overflow
                results.append({
                    "title": d.metadata.get("title", "Wikipedia"),
                    "snippet": content[:200] + "...",
                    "link": d.metadata.get("source", ""),
                    "content": content
                })
            logger.debug(f"Wikipedia titles: {[r['title'] for r in results]}")
            return results
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {str(e)}")
            return []

    def run(self, question: str, question_type: str = "factual") -> str:
        """
        Search Wikipedia for information to answer the question.
        
        Args:
            question: The search question
            question_type: Type of question (kept for compatibility, always uses Wikipedia)
        
        Returns:
            JSON string with search results from Wikipedia
        """
        logger.info(f"Running SearchTool with question_type: {question_type}")
        logger.info(f"Question: {question[:100]}...")
        
        key_terms = self._extract_key_terms(question)
        logger.debug(f"Key terms: {key_terms}")
        
        # Search Wikipedia
        try:
            search_terms = " ".join(key_terms)
            logger.info(f"Searching Wikipedia with terms: {search_terms}")
            all_results = self._search_wikipedia(search_terms)
            logger.info(f"Wikipedia returned {len(all_results)} results")
            if all_results:
                logger.debug(f"Wikipedia titles: {[r['title'] for r in all_results]}")
        except Exception:
            logger.exception("Wikipedia search failed")
            all_results = []

        if not all_results:
            logger.warning("No search results found")
            result = [{"title": "", "snippet": "No search results found.", "link": ""}]
            return json.dumps(result, indent=2)
        
        ranked = self._rank_results(all_results, key_terms)

        if not ranked:
            logger.warning("Ranking produced no results — returning fallback signal.")
            fallback_message = [{
                "title": "No relevant results found",
                "snippet": (
                    "Search did not yield relevant matches. "
                    "Consider rephrasing the query or using another tool such as Wikipedia or general web search."
                ),
                "link": "",
                "content": ""
            }]
            return json.dumps(fallback_message, indent=2)

        logger.debug(f"RAW ranked results: {ranked}")

        top_results = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", ""),
            }
            for r in ranked[: self.max_results]
        ]
        
        logger.info(f"Returning top {len(top_results)} ranked results")
        
        # Fetch full content for top N URLs
        urls = [r["link"] for r in top_results[: self.fetch_top_n] if r.get("link")]
        if urls:
            logger.info(f"Fetching full content from {len(urls)} URLs")
            content_map = self._fetch_full_content(urls)
            
            for r in top_results:
                if r["link"] in content_map:
                    r["content"] = content_map[r["link"]]
        
        logger.info(f"Search complete. Returning {len(top_results)} results")
        return json.dumps(top_results, indent=2)
