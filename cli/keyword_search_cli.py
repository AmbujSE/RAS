# ...existing code...
#!/usr/bin/env python3

import argparse
import json
import string
import pickle
from pathlib import Path
from typing import Optional, Set, List, Dict
from collections import Counter

# try to import NLTK's PorterStemmer, fallback to a no-op stemmer if unavailable
try:
    from nltk.stem import PorterStemmer  # type: ignore
except Exception:

    class PorterStemmer:
        def stem(self, token: str) -> str:
            return token


def load_movies() -> dict:
    movies_path = Path(__file__).resolve().parent.parent / "data" / "movies.json"
    with open(movies_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stopwords() -> Set[str]:
    movies_path = Path(__file__).resolve().parent.parent / "data" / "movies.json"
    stopwords_path = movies_path.parent / "stopwords.txt"
    try:
        with open(stopwords_path, "r", encoding="utf-8") as f:
            return {s.strip().lower() for s in f.read().splitlines() if s.strip()}
    except FileNotFoundError:
        return set()


def normalize_and_tokenize(text: str, translator: dict, stopwords: Set[str], stemmer: Optional[PorterStemmer]) -> List[str]:
    normalized = text.translate(translator).lower()
    tokens = [t for t in normalized.split() if t and t not in stopwords]
    if stemmer:
        return [stemmer.stem(t) for t in tokens]
    return tokens


class InvertedIndex:
    def __init__(self, stemmer: Optional[PorterStemmer] = None, stopwords_set: Optional[Set[str]] = None) -> None:
        self.index: Dict[str, Set[int]] = {}  # token -> set of doc ids
        self.docmap: Dict[int, dict] = {}  # doc id -> document
        self.term_frequencies: Dict[int, Counter] = {}  # doc id -> Counter of tokens
        self.stemmer = stemmer
        self.stopwords_set: Set[str] = stopwords_set or set()
        self.translator = str.maketrans("", "", string.punctuation)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = normalize_and_tokenize(text, self.translator, self.stopwords_set, self.stemmer)
        # update index and term frequencies
        cnt = self.term_frequencies.get(doc_id)
        if cnt is None:
            cnt = Counter()
            self.term_frequencies[doc_id] = cnt
        for tok in tokens:
            # update inverted index
            if tok not in self.index:
                self.index[tok] = set()
            self.index[tok].add(doc_id)
            # update term frequency counter
            cnt[tok] += 1

    def get_documents(self, term: str) -> List[int]:
        term_norm = term.lower()
        if self.stemmer:
            term_norm = self.stemmer.stem(term_norm)
        return sorted(self.index.get(term_norm, set()))

    def get_tf(self, doc_id: str, term: str) -> int:
        # convert doc_id to int if possible
        try:
            doc_id_int = int(doc_id)
        except Exception:
            raise ValueError("doc_id must be convertible to int")
        # tokenize the term (should yield exactly one token)
        normalized = term.translate(self.translator).lower()
        tokens = [t for t in normalized.split() if t and t not in self.stopwords_set]
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        if len(tokens) == 0:
            return 0
        if len(tokens) > 1:
            raise ValueError("term must be a single token after normalization")
        tok = tokens[0]
        return int(self.term_frequencies.get(doc_id_int, Counter()).get(tok, 0))

    def build(self, movies: List[dict]) -> None:
        for m in movies:
            try:
                doc_id = int(m.get("id", 0))
            except Exception:
                doc_id = 0
            self.docmap[doc_id] = m
            text = f"{m.get('title','')} {m.get('description','')}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        cache_dir = Path(__file__).resolve().parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open(cache_dir / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open(cache_dir / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        cache_dir = Path(__file__).resolve().parent.parent / "cache"
        index_path = cache_dir / "index.pkl"
        docmap_path = cache_dir / "docmap.pkl"
        tf_path = cache_dir / "term_frequencies.pkl"
        if not index_path.exists() or not docmap_path.exists() or not tf_path.exists():
            raise FileNotFoundError("Cache files not found")
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index and term frequencies")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                data = load_movies()
            except Exception:
                print("Error: could not load data/movies.json")
                return

            stopwords_set = load_stopwords()
            stemmer = PorterStemmer()
            translator = str.maketrans("", "", string.punctuation)

            # prepare query tokens
            query_normalized = args.query.translate(translator).lower()
            raw_query_tokens = [t for t in query_normalized.split() if t]
            query_tokens = [stemmer.stem(t) for t in raw_query_tokens if t and t not in stopwords_set]

            all_movies = data.get("movies", [])
            results = []

            for m in all_movies:
                title = m.get("title", "") or ""
                description = m.get("description", "") or ""
                text = f"{title} {description}"
                tokens = normalize_and_tokenize(text, translator, stopwords_set, stemmer)
                match_found = any(qt in tt for qt in query_tokens for tt in tokens) if query_tokens else False
                if match_found:
                    results.append(m)

            def _sort_key(item):
                try:
                    return int(item.get("id", 0))
                except Exception:
                    return str(item.get("id", ""))

            results.sort(key=_sort_key)
            results = results[:5]

            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie.get('title','')} ({movie.get('id','')})")
        case "build":
            try:
                data = load_movies()
            except Exception:
                print("Error: could not load data/movies.json")
                return
            stopwords_set = load_stopwords()
            stemmer = PorterStemmer()
            all_movies = data.get("movies", [])
            idx = InvertedIndex(stemmer=stemmer, stopwords_set=stopwords_set)
            idx.build(all_movies)
            idx.save()
            print("Inverted index and term frequencies built and saved to cache.")
        case "tf":
            stopwords_set = load_stopwords()
            stemmer = PorterStemmer()
            idx = InvertedIndex(stemmer=stemmer, stopwords_set=stopwords_set)
            try:
                idx.load()
            except FileNotFoundError:
                print("Error: inverted index cache not found. Run the 'build' command first.")
                return
            try:
                tf_val = idx.get_tf(args.doc_id, args.term)
            except Exception as e:
                print(f"Error: {e}")
                return
            print(tf_val)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
# ...existing code...