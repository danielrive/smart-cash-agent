import json
import spacy
import logging
from typing import List, Dict
from langchain.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import UnstructuredURLLoader

# Module-level logger
logger = logging.getLogger(__name__)


class SearchTool:
    """Search tool with question classification and optimization."""
    logger.info("Initializing SearchTool...")
    def __init__(self, max_results: int = 10,fetch_top_n: int = 3, max_chars: int = 5000):
        self.search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
        self.base_tool = DuckDuckGoSearchResults(api_wrapper=self.search_wrapper, output_format="json")

        self.wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=3000,
            load_all_available_meta=True,
        )

        self.nlp = spacy.load("en_core_web_sm")  # Lightweight NLP model
        self.max_results = max_results

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
        logger.debug("IN Ranking results")
        logger.debug(f"IN RANK RESULT FUNCTION RAW Results: {results}")

        if not results:
            return []
        seen_snippets = set()
        scored = []
        key_terms_lower = [t.lower() for t in key_terms]
        threshold = max(1, len(key_terms_lower) // 2)  

        for r in results:
            snippet = (r.get("snippet", "") + " " + r.get("title", "")).lower()
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            logger.debug(f"Scoring snippet: {snippet}")
            score = sum(snippet.count(term) for term in key_terms_lower)
            logger.debug(f"Score: {score}")
            if len(snippet) < 20:
                score -= 1

            if score >= threshold:
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"IN RANK FUNCTION Scored results: {[scored]}")
        logger.debug(f"IN RANK FUNCTION tittles: {[r['title'] for score, r in scored]}")
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
        
        spicy_doc = self.nlp(question)

        # Step 1: Extract named entities (PERSON, ORG, GPE, WORK_OF_ART, EVENT, etc.)
        entities = [ent.text for ent in spicy_doc.ents if ent.label_ in [
            "PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT"
        ]]
        logger.debug(f"Named entities extracted: {entities}")
        if entities:
            question = " ".join(entities)  # Use entities as the question for Wikipedia search
        try:
            docs = self.wiki_wrapper.load(question)  # returns multiple docs
            logger.debug(f"Wikipedia RAW search results: {docs}")
            results = []
            for d in docs:
                content = d.page_content[:500]  # trim to avoid overflow
                results.append({
                    "title": d.metadata.get("title", "Wikipedia"),
                    "snippet": content[:200] + "...",
                    "link": d.metadata.get("source", ""),
                    "content": content
                })
            return results
        except Exception:
            logger.warning("Wikipedia search failed")
            return []

    def run(self, question: str, question_type: str) -> str:
        logger.info("Running SearchTool")

        logger.info(f"Searching for: {question}")
        
        key_terms = self._extract_key_terms(question)
        all_results = []

        if question_type == "testfactual":
            try:
                search_terms = " ".join(key_terms)
                logger.info(f"Searching Wikipedia for: {search_terms}")
                wiki_results = self._search_wikipedia(search_terms)
                logger.info(f"Wikipedia search results titles: {[r['title'] for r in wiki_results]}")
                logger.debug(f"RAW Wikipedia search results: {wiki_results}")
                all_results.extend(wiki_results)
            except Exception:
                logger.exception("Wikipedia search failed") 

        else:
            try:
                results_raw = self.base_tool.invoke(question)
                
                logger.debug(f"Raw results: {results_raw}")

                if isinstance(results_raw, str):
                    try:
                        results = json.loads(results_raw)
                    except json.JSONDecodeError:
                        logger.exception("Failed to parse results for question")
                        results = []
                    
                elif isinstance(results_raw, list):
                        results = results_raw
                else:
                    results = []

                all_results.extend(results if isinstance(results, list) else [])

            except Exception:
                logger.exception("Web search failed")

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
        
        logger.debug(f"SearchTool ranked results: {top_results}")
        #Fetch full content for top N

        urls = [r["link"] for r in top_results[: self.fetch_top_n]]
        content_map = self._fetch_full_content(urls)

        for r in top_results:
            if r["link"] in content_map:
                r["content"] = content_map[r["link"]]
        logger.debug(f"SearchTool results: {top_results}")
        return json.dumps(top_results, indent=2)
