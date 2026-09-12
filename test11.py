"""
RAG embedding-model comparison, run against a small slice of the real
LegalBench-RAG corpus: Voyage AI (voyage-4-large) vs Cohere (embed-v4.0)
vs a local, self-hosted model (BAAI/bge-large-en-v1.5). 

SETUP BEFORE RUNNING:
1. Download the corpus + benchmarks from the Dropbox link in the LegalBench-RAG
   README and place them at ./data/corpus/ and ./data/benchmarks/
2. pip install voyageai cohere sentence-transformers numpy python-dotenv \
       langchain-text-splitters langchain-core
3. Set your API keys as environment variables:
       export VOYAGE_API_KEY="..."
       export COHERE_API_KEY="..."
4. Edit the CONFIG block below, then run: python run_rag_chunking_eval.py

WHAT THIS DOES:
- Pulls a handful of queries from one benchmark file (e.g. privacy_qa.json),
  constrained so their ground-truth snippets fall within MAX_DOCUMENTS
  distinct source documents (so you're not downloading/chunking/embedding
  the whole corpus for a quick side-by-side comparison).
- Chunks those documents once (chunking is independent of embedding model,
  so all three models are compared on identical chunks).
- Embeds the chunks with each of the three models/providers in MODELS below,
  retrieves the top-k chunks per query, and scores them against the
  benchmark's character-offset ground truth.
- Prints, per query and per model: the file path, the retrieved chunks
  (with a text preview and character-overlap count so you can eyeball answer
  quality), the ground-truth span(s), and character-level precision@k /
  recall@k / F1@k (computed on actual character overlap, not just whether a
  chunk touched a span), plus rank-based hit_rate / MRR.
- Prints a final summary table so you can compare trade-offs across models.

NOTE ON COST: Voyage AI and Cohere are paid APIs billed per token embedded.
With MAX_DOCUMENTS kept small (1-2) and NUM_QUERIES at 5, this script embeds
a small number of chunks, but every run re-embeds them (no caching), so
increase those numbers gradually rather than jumping to the full corpus.
"""

import json
import os
import random
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() 

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

# ============================== CONFIG ==============================
# Point this at ONE benchmark json to start (smaller ones load/run faster).
BENCHMARK_PATH = Path("data/benchmarks/privacy_qa.json")
CORPUS_ROOT = Path("data/corpus")

# How many distinct source documents to load, and how many queries to test.
# Queries are only selected if their ground-truth snippets fit within this
# many documents, so keeping both small keeps the comparison fast and cheap.
MAX_DOCUMENTS = 2
NUM_QUERIES = 5

# Chunking parameters (characters, not tokens — LegalBench-RAG's ground truth
# spans are character offsets, so keeping chunk math in characters avoids a
# token<->character conversion step). Chunking is applied once and shared
# across all models below, so the comparison isolates the embedding model.
CHUNKING_SCHEME = "structure"  # "fixed" or "structure"
FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 100
STRUCTURE_MAX_CHUNK_SIZE = 800
STRUCTURE_CHUNK_OVERLAP = 100

TOP_K = 4  # how many chunks to retrieve per query

# Models to compare. "provider" selects which embed_* function is used.
# Swap model names here if you want e.g. text-embedding-3-small or
# embed-english-v3.0 instead.
MODELS = [
    {"provider": "voyage", "name": "voyage-4-large"},
    {"provider": "cohere", "name": "embed-v4.0"},
    {"provider": "local", "name": "BAAI/bge-large-en-v1.5"},
]

ANSWER_PREVIEW_CHARS = 200  # how much of each retrieved chunk to print
# ======================================================================


# ------------------------------ Chunking ------------------------------

def _split_with_offsets(splitter, text):
    """Run a LangChain text splitter and recover exact character offsets for
    each returned chunk by locating it as a literal substring of `text`,
    searching forward from just past the previous chunk's start.

    We recover offsets this way instead of trusting the splitter's built-in
    `add_start_index` metadata: that feature has a documented bug where it
    returns the wrong start_index whenever a chunk's text recurs elsewhere
    earlier in the document (common in legal boilerplate — repeated defined
    terms, repeated clause headers). Since LegalBench-RAG scoring is entirely
    character-offset-based, a silently wrong offset there would corrupt every
    downstream metric exactly the way the old fixed_chunk bug did. Searching
    forward from the previous match, rather than from position 0 each time,
    sidesteps that failure mode: chunk starts are always monotonically
    increasing, so we never re-match an earlier, wrong occurrence.

    Requires the splitter to be built with strip_whitespace=False so that
    each returned piece is an exact, unmodified substring of `text` and the
    `.find()` lookup below can always succeed.
    """
    pieces = splitter.split_text(text)
    chunks, search_from = [], 0
    for piece in pieces:
        idx = text.find(piece, search_from)
        if idx == -1:
            # Fail loudly rather than silently record a wrong offset.
            raise ValueError(
                "Could not locate a chunk's text back in its source document "
                f"during offset recovery (chunk starts: {piece[:50]!r}). This "
                "should not happen with strip_whitespace=False — check the "
                "splitter configuration."
            )
        start, end = idx, idx + len(piece)
        chunks.append({"text": piece, "start": start, "end": end})
        search_from = start + 1  # forward progress; overlap still allowed
    return chunks


def fixed_chunk(text, chunk_size=FIXED_CHUNK_SIZE, overlap=FIXED_CHUNK_OVERLAP):
    """Split `text` into overlapping windows using LangChain's
    RecursiveCharacterTextSplitter, with exact character offsets recovered
    via _split_with_offsets.

    The old hand-rolled version computed each window's start with pure
    arithmetic (`end - overlap`), which had no relationship to word or
    sentence boundaries and produced chunks starting mid-word. This version
    instead splits by trying progressively finer real separators — paragraph
    break, line break, sentence-ending punctuation, clause comma, then word
    boundary — and only ever falls back to plain character-count splitting
    if a single "word" itself exceeds chunk_size (e.g. a long URL). That
    structurally prevents mid-word/mid-sentence splits rather than papering
    over them after the fact.

    chunk_size / overlap remain character counts, not tokens, matching the
    character-offset ground truth this script evaluates against."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        strip_whitespace=False,
    )
    return _split_with_offsets(splitter, text)


def _looks_like_heading(line):
    """Heuristically identify standalone policy/article headings.

    The LegalBench privacy-policy corpus is not guaranteed to use one heading
    convention. We therefore recognize several common forms instead of relying
    on one regex such as ``1. DEFINITIONS``.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False

    # Remove common heading decoration without destroying the original text.
    cleaned = re.sub(r'^[#>*\-\u2022\s]+', '', stripped)
    cleaned = re.sub(r'[:;,.]+$', '', cleaned).strip()
    if not cleaned:
        return False

    # Numbered headings: 1. Definitions / 1) Definitions / 1.2 Security
    if re.match(r'^\d+(?:[.)]|(?:\.\d+)*\.)\s+\S', cleaned):
        return True

    # Roman/letter section labels: I. Scope / A. Information We Collect
    if re.match(r'^(?:[IVXLCDM]+|[A-Z])[.)]\s+\S', cleaned):
        return True

    words = cleaned.split()
    alpha = re.sub(r'[^A-Za-z]', '', cleaned)
    if not alpha:
        return False

    # Explicit heading-like phrases common in privacy policies.
    heading_terms = {
        'privacy', 'definitions', 'introduction', 'information', 'collection',
        'use', 'disclosure', 'sharing', 'security', 'cookies', 'retention',
        'rights', 'choices', 'access', 'children', 'third-party', 'third party',
        'contact', 'changes', 'updates', 'compliance', 'consent', 'marketing',
        'data', 'personal information', 'your information',
    }
    lower = cleaned.lower()
    if len(words) <= 8 and cleaned.count(',') == 0 and any(
        lower == term or lower.startswith(term + ' ') for term in heading_terms
    ):
        return True

    # ALL-CAPS headings (allow digits/punctuation, require at least one letter).
    letters = [ch for ch in cleaned if ch.isalpha()]
    if letters and sum(ch.isupper() for ch in letters) / len(letters) >= 0.85:
        return len(words) <= 12

    # Short title-like standalone lines: mostly alphabetic, title-cased, and
    # not sentence-like. This catches headings such as "How We Use Information".
    if 1 <= len(words) <= 10 and cleaned[-1] not in '.?!,' and cleaned.count(',') == 0:
        title_like = sum(1 for w in words if w[:1].isupper()) / len(words)
        if title_like >= 0.6:
            return True

    return False


def _structure_boundaries(text):
    """Return (boundaries, method) using the strongest structural signal that
    actually fires, trying progressively weaker signals until one works:

        1. headings (numbered/roman/lettered, ALL-CAPS, title-case lines)
        2. blank-line-separated paragraphs
        3. single-newline lines (docs extracted as one clause/item per line,
           with no blank line between them — common after HTML->text
           conversion of policy pages, which is what tripped this up)
        4. sentence boundaries (tolerating a trailing quote/paren/bracket
           between the period and the next sentence, e.g. `...below." We...`)
        5. genuinely nothing detected: the whole document as one segment

    Only tier 5 should ever fall through to fixed-size splitting inside
    structure_chunk, and the method label lets callers report honestly when
    that's what happened instead of implying paragraph/sentence structure
    was used when it silently wasn't.
    """
    boundaries = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_no_nl = line.rstrip('\r\n')
        if offset > 0 and _looks_like_heading(line_no_nl):
            boundaries.append(offset)
        offset += len(line)

    if boundaries:
        return sorted(set([0, *boundaries, len(text)])), "headings"

    # Tier 2: blank-line paragraphs.
    paragraph_boundaries = [0] + [m.end() for m in re.finditer(r'\n\s*\n+', text)] + [len(text)]
    paragraph_boundaries = sorted(set(paragraph_boundaries))
    if len(paragraph_boundaries) > 2:
        return paragraph_boundaries, "blank-line paragraphs"

    # Tier 3: single-newline lines. Only trust this as real structure (rather
    # than soft-wrapped prose) when most non-blank lines look like complete
    # units — i.e. end in terminal/list punctuation — and lines aren't tiny.
    lines = text.splitlines(keepends=True)
    non_blank = [l.strip() for l in lines if l.strip()]
    if non_blank:
        end_punct_frac = sum(1 for l in non_blank if l[-1] in '.!?:;') / len(non_blank)
        avg_len = sum(len(l) for l in non_blank) / len(non_blank)
        if end_punct_frac >= 0.6 and avg_len >= 30:
            line_boundaries, off = [0], 0
            for line in lines:
                off += len(line)
                if line.strip() and off < len(text):
                    line_boundaries.append(off)
            line_boundaries = sorted(set([*line_boundaries, len(text)]))
            if len(line_boundaries) > 2:
                return line_boundaries, "line breaks"

    # Tier 4: sentence boundaries.
    sentence_boundaries = [0]
    for m in re.finditer(r'(?<=[.!?])["\'\u201d\u2019\)\]]*(?:\s+|\n)(?=["\'\u201c\u2018\(\[]?[A-Z\d])', text):
        sentence_boundaries.append(m.end())
    sentence_boundaries = sorted(set([*sentence_boundaries, len(text)]))
    if len(sentence_boundaries) > 2:
        return sentence_boundaries, "sentence boundaries"

    # Tier 5: nothing structural found anywhere — no headings, no blank
    # lines, no line structure, no detectable sentence punctuation pattern.
    # This is the true last resort; structure_chunk will fixed-chunk this
    # single segment, and callers should say so rather than claiming
    # paragraph/sentence structure was used.
    return [0, len(text)], "none (fixed-size fallback)"


def structure_chunk(text, max_size=STRUCTURE_MAX_CHUNK_SIZE, overlap=STRUCTURE_CHUNK_OVERLAP):
    """Chunk by document structure, then split oversized sections with overlap.

    Unlike the old implementation, a missing ``1. DEFINITIONS``-style header
    does not silently convert the whole document into fixed-size windows.
    We try, in order: heading conventions, blank-line paragraphs, single-line
    breaks (for docs with one clause per line and no blank lines), then
    sentence boundaries — only falling through to fixed-size splitting of the
    whole document if none of those find anything at all. Sections that are
    still oversized after this structural pass are split by fixed_chunk,
    which is itself LangChain-backed and separator-aware (see fixed_chunk's
    docstring) rather than a raw arithmetic window.
    """
    boundaries, _method = _structure_boundaries(text)
    chunks = []

    for s, e in zip(boundaries, boundaries[1:]):
        # Trim only surrounding whitespace for cleaner chunks while preserving
        # the exact character offsets used by LegalBench-RAG evaluation.
        raw_s, raw_e = s, e
        while raw_s < raw_e and text[raw_s].isspace():
            raw_s += 1
        while raw_e > raw_s and text[raw_e - 1].isspace():
            raw_e -= 1
        if raw_s >= raw_e:
            continue

        seg = text[raw_s:raw_e]
        if len(seg) <= max_size:
            chunks.append({"text": seg, "start": raw_s, "end": raw_e})
            continue

        # Oversized structural section: fixed-size windows with overlap, but
        # still local to that section so we never cross unrelated sections.
        for c in fixed_chunk(seg, max_size, overlap):
            chunks.append({
                "text": c["text"],
                "start": raw_s + c["start"],
                "end": raw_s + c["end"],
            })

    return chunks


def chunk_documents(documents, chunk_fn):
    """documents: {file_path: full_text}. Returns {file_path: [chunk dicts]},
    tagging each chunk with its file_path so retrieval can span >1 doc."""
    chunks_by_doc = {}
    for file_path, text in documents.items():
        chunks = chunk_fn(text)
        for c in chunks:
            c["file_path"] = file_path
        chunks_by_doc[file_path] = chunks
    return chunks_by_doc


# --------------------------- Embedding backends ---------------------------
# Each provider's client is created lazily so you don't need e.g. a Cohere
# key just to run the Voyage AI + local comparison.

_voyage_client = None
_cohere_client = None
_local_models = {}


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("Set the VOYAGE_API_KEY environment variable.")
        _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


def _get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        import cohere
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("Set the COHERE_API_KEY environment variable.")
        _cohere_client = cohere.ClientV2(api_key=api_key)
    return _cohere_client


def _get_local_model(name):
    if name not in _local_models:
        from sentence_transformers import SentenceTransformer
        _local_models[name] = SentenceTransformer(name, trust_remote_code=True)
    return _local_models[name]


def _batched(items, size=90):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def embed_voyage(texts, model_name, input_type):
    """Embed texts with Voyage AI using document/query-specific input types."""
    client = _get_voyage_client()
    vectors = []
    for batch in _batched(texts):
        resp = client.embed(
            batch,
            model=model_name,
            input_type=input_type,
            output_dtype="float",
        )
        vectors.extend(resp.embeddings)
    arr = np.array(vectors, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)  # defensive normalize
    return arr


def embed_cohere(texts, model_name, input_type):
    client = _get_cohere_client()
    vectors = []
    for batch in _batched(texts):
        resp = client.embed(
            texts=batch,
            model=model_name,
            input_type=input_type,
            embedding_types=["float"],
        )
        vectors.extend(resp.embeddings.float)
    arr = np.array(vectors, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def embed_local(texts, model_name):
    model = _get_local_model(model_name)
    arr = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(arr, dtype=np.float32)


def _embed_chunks_raw(provider, model_name, chunks):
    """Embeds chunk text with the given provider/model. Returns an
    (n_chunks, dim) array aligned with `chunks`. This is the same
    per-provider dispatch as before (renamed with a leading underscore) —
    ProviderEmbeddings below is a thin interface wrapper around it, not a
    replacement for it."""
    texts = [c["text"] for c in chunks]
    if provider == "voyage":
        return embed_voyage(texts, model_name, input_type="document")
    if provider == "cohere":
        return embed_cohere(texts, model_name, input_type="search_document")
    if provider == "local":
        return embed_local(texts, model_name)
    raise ValueError(f"Unknown provider: {provider}")


def _embed_query_raw(provider, model_name, query):
    if provider == "voyage":
        return embed_voyage([query], model_name, input_type="query")[0]
    if provider == "cohere":
        return embed_cohere([query], model_name, input_type="search_query")[0]
    if provider == "local":
        return embed_local([query], model_name)[0]
    raise ValueError(f"Unknown provider: {provider}")


class ProviderEmbeddings(Embeddings):
    """LangChain `Embeddings` adapter around this script's existing
    embed_voyage / embed_cohere / embed_local functions.

    This is intentionally a thin pass-through, not a reimplementation: the
    provider-specific API calls, batching (_batched), input_type selection
    (document vs query, and each provider's own naming for that), and
    normalization all stay exactly as they were in _embed_chunks_raw /
    _embed_query_raw. Wrapping them here only gives the embedding step a
    standard interface — useful if you plug in a LangChain VectorStore later
    (deliverable 2's /search endpoint) — without changing which model does
    the embedding or how its output is computed.
    """

    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name

    def embed_documents(self, texts):
        pseudo_chunks = [{"text": t} for t in texts]
        return _embed_chunks_raw(self.provider, self.model_name, pseudo_chunks).tolist()

    def embed_query(self, text):
        return _embed_query_raw(self.provider, self.model_name, text).tolist()


# ------------------------------ Evaluation ------------------------------
# All scoring below is at the CHARACTER level, not the chunk level: we treat
# retrieved chunks and ground-truth snippets as sets of character positions
# (per document) and score the actual overlap in characters, not just
# whether a chunk happened to touch a span at all. This matters because a
# chunk that overlaps 3 characters of a 400-character ground-truth span and
# a chunk that overlaps all 400 are very different retrievals, and chunk-
# level hit/miss scoring can't tell them apart.

def merge_intervals(intervals):
    """intervals: list of (start, end) tuples, same document. Returns them
    sorted and merged so overlapping/adjacent ranges don't get double-counted."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def interval_set_len(intervals):
    """Total character count covered by a list of (already merged, but works
    either way since it just sums) non-overlapping intervals."""
    return sum(e - s for s, e in intervals)


def intersect_intervals(a, b):
    """a, b: merged, sorted (start, end) interval lists. Returns their
    overlapping regions as a new interval list."""
    i, j, out = 0, 0, []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _merged_intervals_by_file(items):
    """items: list of {"file_path","start","end"} (chunks or gt spans).
    Returns {file_path: merged interval list}."""
    by_file = {}
    for it in items:
        by_file.setdefault(it["file_path"], []).append((it["start"], it["end"]))
    return {fp: merge_intervals(ivs) for fp, ivs in by_file.items()}


def evaluate_query(retrieved_chunks, ground_truth_spans, k=TOP_K):
    """retrieved_chunks: top-k chunks for one query, already ranked.
    ground_truth_spans: list of {"file_path","start","end"} — every snippet
    the benchmark marks as relevant for this query.

    Precision/recall/F1 are computed on character overlap (unioned across
    chunks/spans within a document, so overlapping chunks don't inflate the
    count). hit_rate/mrr stay rank-based (did/when did we retrieve *any*
    overlapping character) since those are inherently about ranking, not
    overlap magnitude.
    """
    top_k = retrieved_chunks[:k]

    retrieved_by_file = _merged_intervals_by_file(top_k)
    gt_by_file = _merged_intervals_by_file(ground_truth_spans)

    retrieved_chars = sum(interval_set_len(ivs) for ivs in retrieved_by_file.values())
    gt_chars = sum(interval_set_len(ivs) for ivs in gt_by_file.values())

    overlap_chars = sum(
        interval_set_len(intersect_intervals(retrieved_by_file.get(fp, []), gt_ivs))
        for fp, gt_ivs in gt_by_file.items()
    )

    char_precision = overlap_chars / retrieved_chars if retrieved_chars else 0.0
    char_recall = overlap_chars / gt_chars if gt_chars else 0.0
    char_f1 = (2 * char_precision * char_recall / (char_precision + char_recall)
               if (char_precision + char_recall) else 0.0)

    # Rank-based metrics: a chunk "hits" if it has ANY character overlap
    # with the ground truth in its own document.
    hits = []
    for c in top_k:
        c_gt = gt_by_file.get(c["file_path"], [])
        overlap = interval_set_len(intersect_intervals(merge_intervals([(c["start"], c["end"])]), c_gt))
        hits.append(overlap > 0)
    hit_rate = 1.0 if any(hits) else 0.0
    mrr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)

    return {
        "char_precision@k": round(char_precision, 4),
        "char_recall@k": round(char_recall, 4),
        "char_f1@k": round(char_f1, 4),
        "hit_rate": hit_rate,
        "mrr": mrr,
        "overlap_chars": overlap_chars,
        "retrieved_chars": retrieved_chars,
        "gt_chars": gt_chars,
    }


# ------------------------------ Loading ------------------------------

def load_benchmark(benchmark_path, corpus_root, max_documents, num_queries, shuffle_seed=0):
    """Walks the benchmark's tests in randomized order, greedily keeping a
    query only if it doesn't push the running set of distinct source
    documents past max_documents. Stops once num_queries queries are collected.

    The benchmark file groups all queries for one document together before
    moving to the next, so walking it in original order exhausts num_queries
    within the first document or two and MAX_DOCUMENTS never binds. Shuffling
    with a fixed seed spreads document coverage while staying reproducible.

    Adjust the key names below (test['query'], snippet['file_path'],
    snippet['span']) if your downloaded json uses different field names —
    print(raw[0]) after loading to check before running the full thing.
    """
    raw = json.loads(benchmark_path.read_text())
    tests = raw["tests"] if "tests" in raw else raw
    tests = list(tests)
    random.Random(shuffle_seed).shuffle(tests)

    selected_docs = []      # unique file_paths, in first-seen (post-shuffle) order
    selected_queries = []   # [{"query": str, "spans": [{"file_path","start","end"}]}]
    missing_files = set()   # corpus files referenced by the benchmark but absent on disk

    for test in tests:
        if len(selected_queries) >= num_queries:
            break
        snippet_docs = {s["file_path"] for s in test["snippets"]}
        new_docs = snippet_docs - set(selected_docs)
        if len(selected_docs) + len(new_docs) > max_documents:
            continue  # would need more documents than we're allowed — skip
        unavailable = {d for d in new_docs if not (corpus_root / d).exists()}
        if unavailable:
            # A referenced corpus file doesn't exist (missing download, or a
            # filename mismatch — e.g. a colon in the benchmark's file_path
            # that got sanitized during extraction). Skip this test entirely
            # rather than counting the doc toward the budget and leaving a
            # query pointing at a document that never gets loaded — that
            # would KeyError deep in the retrieval loop instead of here.
            missing_files |= unavailable
            continue
        selected_docs.extend(sorted(new_docs))
        spans = [{"file_path": s["file_path"], "start": s["span"][0], "end": s["span"][1]}
                 for s in test["snippets"]]
        selected_queries.append({"query": test["query"], "spans": spans})

    for f in sorted(missing_files):
        print(f"  [warn] corpus file not found, skipped every query needing it: {corpus_root / f}")

    documents = {
        file_path: (corpus_root / file_path).read_text() for file_path in selected_docs
    }

    return documents, selected_queries


# ------------------------------ Retrieval ------------------------------
# Plain top-k similarity search via a LangChain VectorStore. Each model gets
# its own InMemoryVectorStore, built once (all chunks embedded a single time
# via add_texts, using the same ProviderEmbeddings adapter as before) and
# reused across all queries in the run. Per query, retrieval is scoped to
# only the documents referenced by that query's ground truth (ref_docs) via
# the store's `filter` argument, matching the previous combined_chunks
# scoping, but the ranking itself is now the vector store's own top-k
# similarity search — no overlap-aware re-ranking on top of it.

def build_vector_store(embeddings_model, chunks_by_doc):
    """Embeds every chunk across all documents once and returns an
    InMemoryVectorStore over them, with each chunk's file_path/start/end
    carried as metadata so results can be mapped back to source offsets."""
    vector_store = InMemoryVectorStore(embeddings_model)
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    vector_store.add_texts(
        texts=[c["text"] for c in all_chunks],
        metadatas=[
            {"file_path": c["file_path"], "start": c["start"], "end": c["end"]}
            for c in all_chunks
        ],
    )
    return vector_store


def _doc_to_chunk(doc):
    return {
        "text": doc.page_content,
        "file_path": doc.metadata["file_path"],
        "start": doc.metadata["start"],
        "end": doc.metadata["end"],
    }


# -------------------------------- Run --------------------------------

def run():
    documents, selected_queries = load_benchmark(
        BENCHMARK_PATH, CORPUS_ROOT, MAX_DOCUMENTS, NUM_QUERIES
    )
    print(f"Loaded {len(documents)} document(s), {len(selected_queries)} quer{'y' if len(selected_queries) == 1 else 'ies'}.")
    for fp in documents:
        print(f"  - {fp}")

    chunk_fn = {"fixed": fixed_chunk, "structure": structure_chunk}[CHUNKING_SCHEME]
    chunks_by_doc = chunk_documents(documents, chunk_fn)
    for fp, chunks in chunks_by_doc.items():
        print(f"  {fp}: {len(chunks)} chunks ({CHUNKING_SCHEME} chunking)")
        if CHUNKING_SCHEME == "structure":
            struct_boundaries, method = _structure_boundaries(documents[fp])
            n_boundaries = max(0, len(struct_boundaries) - 2)
            print(f"    {n_boundaries} structural boundaries detected (method: {method})")
            if method == "none (fixed-size fallback)":
                print("    [warn] No headings, blank-line paragraphs, line breaks, or sentence "
                      "punctuation patterns were found in this document at all — it's being "
                      "split with the fixed-size fallback inside structure_chunk. Inspect the "
                      "raw extracted text for this file; this is the one case where 'structure' "
                      "chunking is genuinely no better than 'fixed'.")

    summary = {}

    for model_cfg in MODELS:
        provider, model_name = model_cfg["provider"], model_cfg["name"]
        label = f"{provider}/{model_name}"
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

        try:
            embeddings_model = ProviderEmbeddings(provider, model_name)
            # Embeds every chunk once (single add_texts call) and reuses this
            # store across all queries below — no re-embedding per query.
            vector_store = build_vector_store(embeddings_model, chunks_by_doc)

            all_metrics = {"char_precision@k": [], "char_recall@k": [], "char_f1@k": [],
                           "hit_rate": [], "mrr": []}

            for q in selected_queries:
                query_text = q["query"]
                gt_spans = q["spans"]
                ref_docs = sorted({s["file_path"] for s in gt_spans})
                ref_docs_set = set(ref_docs)
                total_candidates = sum(len(chunks_by_doc[fp]) for fp in ref_docs)

                # Plain top-k similarity search via the vector store, scoped to
                # the documents referenced by this query's ground truth. Asking
                # for all matching candidates (not just TOP_K) gives us the full
                # ranking for the diagnostics below at no extra embedding cost —
                # it's the same query vector, just a bigger k on the same store.
                ranked = vector_store.similarity_search_with_score(
                    query_text,
                    k=total_candidates,
                    filter=lambda doc, _refs=ref_docs_set: doc.metadata["file_path"] in _refs,
                )
                full_order_chunks = [_doc_to_chunk(doc) for doc, _ in ranked]
                full_order_scores = [score for _, score in ranked]
                retrieved = full_order_chunks[:TOP_K]

                # Diagnostic: rank of the first chunk (in the FULL ranking, not
                # just top-k) that overlaps the ground truth at all. Distinguishes
                # "the answer chunk exists but ranked below top-k" (a retrieval/
                # ranking issue) from "no chunk in the pool overlaps the ground
                # truth at all" (a chunking issue — the answer got cut apart or
                # was never captured as a coherent chunk to begin with).
                gt_by_file_full = _merged_intervals_by_file(gt_spans)
                first_hit_rank = None
                for rank, c in enumerate(full_order_chunks, start=1):
                    c_gt = gt_by_file_full.get(c["file_path"], [])
                    if interval_set_len(intersect_intervals(
                            merge_intervals([(c["start"], c["end"])]), c_gt)) > 0:
                        first_hit_rank = rank
                        break

                metrics = evaluate_query(retrieved, gt_spans, k=TOP_K)
                for m in all_metrics:
                    all_metrics[m].append(metrics[m])

                gt_by_file = _merged_intervals_by_file(gt_spans)

                print(f"\n  Query: {query_text}")
                print(f"  Ground truth: {[(s['file_path'], s['start'], s['end']) for s in gt_spans]}"
                      f" ({metrics['gt_chars']} chars total)")
                if first_hit_rank is None:
                    print(f"  [diagnostic] NO chunk among all {len(full_order_chunks)} candidate "
                          f"chunks overlaps the ground truth at all — the answer text isn't "
                          f"captured by any generated chunk.")
                elif first_hit_rank > TOP_K:
                    print(f"  [diagnostic] first semantic hit ranked #{first_hit_rank} of "
                          f"{len(full_order_chunks)}; plain top_k={TOP_K} retrieval missed it.")
                for rank, c in enumerate(retrieved, start=1):
                    c_gt = gt_by_file.get(c["file_path"], [])
                    c_overlap = interval_set_len(intersect_intervals(
                        merge_intervals([(c["start"], c["end"])]), c_gt))
                    preview = c["text"][:ANSWER_PREVIEW_CHARS].replace("\n", " ").strip()
                    print(f"    [{rank}] {c_overlap:4d} chars overlap  "
                          f"{c['file_path']} [{c['start']}:{c['end']}] score={full_order_scores[rank - 1]:.3f}")
                    print(f"         {preview}...")
                print(f"  Metrics: {metrics}")

            avg = {m: round(sum(v) / len(v), 3) for m, v in all_metrics.items()}
            print(f"\n  --- {label} averages over {len(all_metrics['hit_rate'])} queries: {avg} ---")
            summary[label] = avg

        except Exception as e:
            # Don't let one provider's failure (bad/expired key, no credits,
            # rate limit, network issue, etc.) take down the whole comparison
            # — report it and move on to the next model.
            print(f"\n  [SKIPPED] {label} failed: {type(e).__name__}: {e}")
            continue

    print(f"\n{'=' * 60}\nSUMMARY (avg over all {len(selected_queries)} queries)\n{'=' * 60}")
    print(f"{'model':40s} {'char_prec@k':>11s} {'char_rec@k':>11s} {'char_f1@k':>10s} "
          f"{'hit_rate':>9s} {'mrr':>6s}")
    for label, avg in summary.items():
        print(f"{label:40s} {avg['char_precision@k']:11.3f} {avg['char_recall@k']:11.3f} "
              f"{avg['char_f1@k']:10.3f} {avg['hit_rate']:9.3f} {avg['mrr']:6.3f}")


if __name__ == "__main__":
    run()