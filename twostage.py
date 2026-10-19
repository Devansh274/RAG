"""
RAG embedding-model comparison, run against a small slice of the real
LegalBench-RAG corpus: Voyage AI (voyage-4-large) vs Cohere (embed-v4.0)
vs a local, self-hosted model (BAAI/bge-large-en-v1.5). 

SETUP BEFORE RUNNING:
1. Download the corpus + benchmarks from the Dropbox link in the LegalBench-RAG
   README and place them at ./data/corpus/ and ./data/benchmarks/
2. pip install voyageai cohere sentence-transformers numpy python-dotenv \
       langchain-text-splitters langchain-core anthropic google-genai
3. Set your API keys as environment variables:
       export VOYAGE_API_KEY="..."
       export COHERE_API_KEY="..."
       export ANTHROPIC_API_KEY="..."   # only needed if RERANK_ENABLED = True and
                                         # RERANK_PROVIDER = "anthropic"
       export GEMINI_API_KEY="..."      # only needed if RERANK_ENABLED = True and
                                         # RERANK_PROVIDER = "gemini"
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
- If RERANK_ENABLED, ALSO reranks: pulls a larger cosine-similarity candidate
  pool per query, asks an LLM (Claude) to score each candidate's relevance to
  the query directly, re-sorts by that score, and scores the new top-k the
  same way. This runs as a second, separately-reported variant of each
  embedding model ("<model> + LLM rerank") so you can see whether reranking
  actually helped, on this corpus, for that model -- not just assume it does.
- TWO-STAGE RETRIEVAL (METADATA_RETRIEVAL_ENABLED): LegalBench-RAG queries
  look like "<hint naming the document>; <the actual question>". Stage 1
  (metadata retrieval) uses the part BEFORE the first ';' to pick a small
  subset of candidate documents. Stage 2 (similarity search) embeds only the
  part AFTER the ';' and ranks chunks from that subset alone. Set the flag to
  False to get the original single-stage, whole-corpus search for comparison.
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
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() 

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

# ============================== CONFIG ==============================
# Point this at ONE benchmark json to start (smaller ones load/run faster).
BENCHMARK_PATH = Path("data/benchmarks/cuad.json")
CORPUS_ROOT = Path("data/corpus")

# How many distinct source documents to load, and how many queries to test.
# Queries are only selected if their ground-truth snippets fit within this
# many documents, so keeping both small keeps the comparison fast and cheap.
#
# Because BENCHMARK_PATH above points at cuad.json only, load_benchmark()
# walks CUAD's tests exclusively -- every selected query comes from CUAD,
# and every document it pulls in comes from each query's snippet file_paths,
# which are themselves CUAD corpus files (e.g. "cuad/<contract>.txt" under
# CORPUS_ROOT). So pointing BENCHMARK_PATH at cuad.json is sufficient to
# constrain both documents and queries to CUAD -- no change to
# load_benchmark's selection logic itself is needed.
MAX_DOCUMENTS = 10
NUM_QUERIES = 80

USE_FIXED_DOCUMENT_SUBSET = True
FIXED_DOCUMENT_SUBDIR = "cuad"

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

# --- Two-stage retrieval: (1) metadata -> document subset, (2) similarity ---
# Queries are "<document hint>; <question>". Stage 1 ranks DOCUMENTS against
# the hint (the text before the first separator) and keeps a small subset;
# stage 2 runs the usual chunk-level cosine search, using only the question
# (the text after the separator), restricted to chunks of that subset.
# If a query has no separator (or an empty side), it falls back to searching
# every document with the full query text, so it is never dropped.
# Set METADATA_RETRIEVAL_ENABLED = False to reproduce the old behaviour
# exactly (full query, whole corpus) -- handy as a baseline run.
METADATA_RETRIEVAL_ENABLED = True
METADATA_QUERY_SEPARATOR = ";"
# How documents are matched against the hint:
#   "bm25"      lexical match on file name + document header. Free, no API
#               calls, strong when the hint repeats party/policy names.
#   "embedding" cosine similarity between the hint and each document's
#               name+header, using the SAME embedding model being evaluated
#               (costs one embedding per document + one per query).
#   "hybrid"    reciprocal-rank fusion of the two above.
METADATA_METHOD = "bm25"
METADATA_TOP_DOCS = 3          # max documents kept after stage 1
METADATA_MIN_SCORE_RATIO = 0.5 # bm25 only: also drop docs scoring below this
                               # fraction of the best doc's score (0 = off)
METADATA_HEADER_CHARS = 1500   # first N chars of each doc used as its "header"
METADATA_FILENAME_WEIGHT = 3   # file-name tokens count this many times in BM25

# Models to compare. "provider" selects which embed_* function is used.
# Swap model names here if you want e.g. text-embedding-3-small or
# embed-english-v3.0 instead.
MODELS = [
    {"provider": "voyage", "name": "voyage-4-large"},
    {"provider": "cohere", "name": "embed-v4.0"},
    # {"provider": "local", "name": "BAAI/bge-large-en-v1.5"},
]

ANSWER_PREVIEW_CHARS = 650  # how much of each retrieved chunk to print

# --- LLM-based reranking (optional second stage on top of vector search) ---
# When enabled, retrieval per query becomes two-stage:
#   1. Plain cosine similarity (the existing vector store search) pulls a
#      larger candidate pool of RERANK_CANDIDATE_POOL chunks.
#   2. An LLM (Claude) is shown the query plus every candidate and scores
#      each one's relevance directly -- not via embedding distance -- and
#      the pool is re-sorted by that score before taking the top TOP_K.
# This is reported as a SEPARATE row per embedding model ("<model> + LLM
# rerank") alongside the plain cosine-only row, so the summary table shows
# directly whether reranking helped, hurt, or made no measurable difference
# for each embedding model on this corpus -- rather than assuming it does.
RERANK_ENABLED = False
RERANK_PROVIDER = "gemini"          # "anthropic" or "gemini"
RERANK_MODEL = "gemini-3.5-flash-lite"   # requires GEMINI_API_KEY to be set
                                     # (anthropic alternative: "claude-sonnet-5",
                                     # requires ANTHROPIC_API_KEY instead)
RERANK_CANDIDATE_POOL = 20          # how many cosine-ranked chunks to hand to the LLM
RERANK_MAX_CHARS_PER_CHUNK = 650    # truncate chunk text in the rerank prompt (cost control)
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
        # 'privacy', 'definitions', 'introduction', 'information', 'collection',
        # 'use', 'disclosure', 'sharing', 'security', 'cookies', 'retention',
        # 'rights', 'choices', 'access', 'children', 'third-party', 'third party',
        # 'contact', 'changes', 'updates', 'compliance', 'consent', 'marketing',
        # 'data', 'personal information', 'your information',
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
_anthropic_client = None
_gemini_client = None


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


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set the GEMINI_API_KEY environment variable.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _batched(items, size=90):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def _is_retryable_error(exc):
    """True if exc looks like a transient error worth retrying: a 429 rate/
    quota limit, or the provider's servers being temporarily overloaded
    (503/500/529 -- e.g. Gemini's "high demand" ServerError, Anthropic's
    "overloaded_error"). voyageai, cohere, anthropic, and google-genai each
    raise their own exception types, so check status_code/code first and
    fall back to the exception's class name / message for SDK versions that
    only surface it that way.

    Note this does NOT distinguish a per-minute rate limit (worth retrying)
    from a per-day quota exhausted (retrying won't help until it resets) --
    both surface as 429s. If you see repeated retries ending in a 429 whose
    message mentions a daily/RPD quota, the fix is a model with a larger
    free-tier allowance or enabling billing, not waiting longer.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)  # google-genai's ClientError/ServerError use .code
    if status in (429, 500, 503, 529):
        return True
    name = type(exc).__name__.lower()
    if any(s in name for s in ("toomanyrequests", "ratelimit", "overloaded",
                                "unavailable", "serviceunavailable")):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in ("429", "rate limit", "503", "unavailable", "overloaded"))


def _call_with_backoff(fn, *args, max_retries=6, base_delay=20, **kwargs):
    """Calls fn(*args, **kwargs), retrying with exponential backoff ONLY on
    retryable errors (see _is_retryable_error) -- anything else is raised
    immediately. Needed because a provider's per-minute token limit (or a
    trial key's stricter limits) can be hit even when nothing about the
    request is wrong, and because providers' servers do occasionally return
    a transient 503 under load; the right response to either is to wait and
    retry, not to fail the whole run."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable_error(exc) or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"    [rate limited] retrying in {delay}s "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)


def embed_voyage(texts, model_name, input_type):
    """Embed texts with Voyage AI using document/query-specific input types."""
    client = _get_voyage_client()
    vectors = []
    for batch in _batched(texts):
        resp = _call_with_backoff(
            client.embed,
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
        resp = _call_with_backoff(
            client.embed,
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


def _embed_queries_raw(provider, model_name, queries):
    """Embeds every query in `queries` in one pass, batched the same way
    _embed_chunks_raw batches documents (via embed_voyage/embed_cohere's own
    _batched loop). Used to embed all selected queries up front instead of
    calling _embed_query_raw once per query inside the per-query eval loop
    -- that per-query pattern means one network call per query (80 calls
    for NUM_QUERIES=80), which burns through a provider's per-minute token
    budget and, for a trial API key, its total call quota, much faster than
    a handful of batched calls covering the same queries."""
    if provider == "voyage":
        return embed_voyage(queries, model_name, input_type="query")
    if provider == "cohere":
        return embed_cohere(queries, model_name, input_type="search_query")
    if provider == "local":
        return embed_local(queries, model_name)
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


# ---------------------------- LLM Reranking ----------------------------
# Second-stage reranker: given a query and a pool of chunks already ranked
# by cosine similarity, ask an LLM to judge each chunk's relevance to the
# query directly (reading the actual text, not comparing embedding vectors)
# and re-sort by that judgment. This is a classic "cross-encoder"-style
# reranking step -- the LLM sees the query and the passage together in one
# prompt, which lets it catch relevance the embedding model missed (e.g. a
# chunk that's topically similar but doesn't answer the specific question
# asked, or the reverse: a chunk that's lexically distant from the query but
# is clearly the right answer once you actually read it).
#
# This is done LISTWISE, one LLM call per query covering the whole
# candidate pool, rather than POINTWISE (a separate call per chunk asking
# "how relevant is this one"). Pointwise is simpler to reason about and
# slightly more robust to any one bad score, but for NUM_QUERIES=80 and a
# ~20-chunk pool it's 1,600 LLM calls instead of 80 -- roughly 20x the cost
# and latency for a benchmark run. Listwise scoring also lets the model
# weigh candidates against each other, not just against the query in
# isolation, which tends to help it calibrate scores consistently.

def _parse_rerank_scores(raw_text, n_chunks):
    """Parse the LLM's response into a list of n_chunks scores, indexed to
    match the order chunks were shown in the prompt. Tolerant of a ```json
    fence around the array (some models add one despite being told not to).

    Any index the model skipped, duplicated, or returned out of range keeps
    a fallback score instead of raising: real scores win, and any unscored
    chunk is placed just below the lowest real score, in its original
    cosine-rank order relative to other unscored chunks. That way a partial
    or slightly malformed response still produces a usable ranking (real
    judgments first, un-judged chunks falling back to cosine order) instead
    of losing the whole query to a formatting glitch.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    scores = [None] * n_chunks
    try:
        parsed = json.loads(cleaned)
        for item in parsed:
            idx = int(item["index"])
            if 0 <= idx < n_chunks and scores[idx] is None:
                scores[idx] = float(item["score"])
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        pass  # everything stays None -> all chunks get the fallback below

    real_scores = [s for s in scores if s is not None]
    floor = (min(real_scores) - 1) if real_scores else 0.0
    for i in range(n_chunks):
        if scores[i] is None:
            scores[i] = floor - (i * 1e-6)  # preserves original relative order
    return scores


def _build_rerank_prompt(query, chunks, max_chars):
    passages_block = "\n\n".join(
        f"[{i}] {c['text'][:max_chars]}" for i, c in enumerate(chunks)
    )
    return (
        "You are scoring passages from legal documents for relevance to a search "
        "query, to rerank retrieval results for a legal RAG system.\n\n"
        "Score each passage from 0 to 10:\n"
        "- 0-2: irrelevant, or only shares surface vocabulary with the query\n"
        "- 3-5: topically related but does not answer what the query is asking\n"
        "- 6-8: partially answers the query, or contains relevant but incomplete info\n"
        "- 9-10: directly contains the specific clause/fact/answer the query needs\n\n"
        f"Query: {query}\n\n"
        f"Passages:\n{passages_block}\n\n"
        "Respond with ONLY a JSON array, no other text before or after it, of "
        'objects {"index": <passage number>, "score": <0-10>}, one entry per '
        "passage, covering every passage index exactly once."
    )


def _rerank_call_anthropic(prompt, model_name):
    """Returns the model's raw text response for the rerank prompt."""
    client = _get_anthropic_client()
    resp = _call_with_backoff(
        client.messages.create,
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _rerank_call_gemini(prompt, model_name):
    """Returns the model's raw text response for the rerank prompt.

    Sets response_mime_type="application/json" (Gemini-specific constrained
    decoding), which is stricter than just asking nicely in the prompt text
    and makes the JSON-parsing fallback in _parse_rerank_scores a rarely-hit
    safety net rather than the norm.
    """
    from google.genai import types
    client = _get_gemini_client()
    resp = _call_with_backoff(
        client.models.generate_content,
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    return resp.text or ""


_RERANK_CALL_FNS = {
    "anthropic": _rerank_call_anthropic,
    "gemini": _rerank_call_gemini,
}


def llm_rerank(query, chunks, provider=RERANK_PROVIDER, model_name=RERANK_MODEL,
                max_chars=RERANK_MAX_CHARS_PER_CHUNK):
    """chunks: a candidate pool (list of chunk dicts), already in cosine-
    similarity order. Returns (reranked_chunks, scores) -- the same chunks,
    reordered by descending LLM-judged relevance, and the score assigned to
    each (aligned to the returned order, not the input order).

    `provider` selects which underlying API does the scoring; the prompt,
    parsing, and everything else here are shared across providers -- only
    the raw API call (_rerank_call_anthropic / _rerank_call_gemini) differs.
    """
    if not chunks:
        return [], []
    if provider not in _RERANK_CALL_FNS:
        raise ValueError(f"Unknown RERANK_PROVIDER: {provider!r}")

    prompt = _build_rerank_prompt(query, chunks, max_chars)
    raw_text = _RERANK_CALL_FNS[provider](prompt, model_name)
    scores = _parse_rerank_scores(raw_text, len(chunks))

    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in order], [scores[i] for i in order]


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
    top1_hit_rate = 1.0 if hits and hits[0] else 0.0   # NEW: rank-1 chunk overlaps ground truth

    return {
        "char_precision@k": round(char_precision, 4),
        "char_recall@k": round(char_recall, 4),
        "char_f1@k": round(char_f1, 4),
        "hit_rate": hit_rate,
        "top1_hit_rate": top1_hit_rate, 
        "mrr": mrr,
        "overlap_chars": overlap_chars,
        "retrieved_chars": retrieved_chars,
        "gt_chars": gt_chars,
    }

def micro_prf(overlap_chars, retrieved_chars, gt_chars):
    """Micro-averaged P/R/F1 from character counts summed over all queries."""
    p = overlap_chars / retrieved_chars if retrieved_chars else 0.0
    r = overlap_chars / gt_chars if gt_chars else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"micro_precision@k": round(p, 3),
            "micro_recall@k": round(r, 3),
            "micro_f1@k": round(f1, 3)}

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
    raw = json.loads(benchmark_path.read_text(encoding="utf-8"))
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
        file_path: (corpus_root / file_path).read_text(encoding="utf-8") for file_path in selected_docs
    }

    return documents, selected_queries

def list_first_n_documents(corpus_root, subdir, n):
    """Returns the first n document file_paths (e.g. "cuad/<name>.txt",
    relative to corpus_root -- the same form used in benchmark json and in
    chunks_by_doc / gt_spans elsewhere) from corpus_root/subdir, sorted by
    filename. Sorted-filename order is arbitrary but deterministic and
    reproducible, unlike load_benchmark's shuffle-driven selection."""
    files = sorted((corpus_root / subdir).glob("*.txt"))
    return [f"{subdir}/{f.name}" for f in files[:n]]


def load_benchmark_fixed_docs(benchmark_path, corpus_root, doc_subset, num_queries):
    """Like load_benchmark, but the document set is FIXED (doc_subset) rather
    than grown greedily to fit queries. Walks the benchmark's tests in
    their original order and keeps a query only if EVERY one of its ground
    truth snippets falls inside doc_subset -- a query that needs even one
    document outside the subset is skipped entirely, never truncated to
    its in-subset snippets, since that would silently understate its true
    ground truth. Stops once num_queries queries are collected.

    Pair with list_first_n_documents() to get "first N documents" behavior;
    doc_subset can also be any explicit list of file_paths you want to test.
    """
    raw = json.loads(benchmark_path.read_text(encoding="utf-8"))
    tests = raw["tests"] if "tests" in raw else raw
    doc_subset = set(doc_subset)

    selected_queries = []
    for test in tests:
        if len(selected_queries) >= num_queries:
            break
        snippet_docs = {s["file_path"] for s in test["snippets"]}
        if not snippet_docs <= doc_subset:
            continue  # needs a document outside the fixed subset -- skip, don't truncate
        spans = [{"file_path": s["file_path"], "start": s["span"][0], "end": s["span"][1]}
                 for s in test["snippets"]]
        selected_queries.append({"query": test["query"], "spans": spans})

    missing = {d for d in doc_subset if not (corpus_root / d).exists()}
    for f in sorted(missing):
        print(f"  [warn] corpus file not found: {corpus_root / f}")
    documents = {
        d: (corpus_root / d).read_text(encoding="utf-8")
        for d in sorted(doc_subset) if d not in missing
    }
    if not selected_queries:
        print(f"  [warn] no queries found whose ground truth lies entirely within "
              f"the {len(doc_subset)} fixed documents -- try more documents.")
    return documents, selected_queries


# ------------------------------ Retrieval ------------------------------
# Retrieval is two-stage (see METADATA_RETRIEVAL_ENABLED):
#   Stage 1 -- metadata retrieval: DocumentMetadataIndex ranks documents
#     against the hint part of the query and returns a small subset.
#   Stage 2 -- similarity search: plain top-k cosine search via a LangChain
#     VectorStore, using the question part of the query, scoped to the stage-1
#     subset via the store's `filter` argument.
# Each model gets its own InMemoryVectorStore, built once (all chunks embedded
# a single time via add_texts, using the same ProviderEmbeddings adapter as
# before) and reused across all queries in the run. The ranking itself is the
# vector store's own top-k similarity search -- no overlap-aware re-ranking.
#
# Unlike an earlier version of this script, the subset is NOT derived from
# the ground-truth spans (that leaked the answer). It comes only from the
# query's own hint text, so stage-1 misses are real and are reported.

_METADATA_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "by", "with",
    "between", "among", "from", "at", "as", "is", "are", "s",
    "consider", "regarding", "about", "under", "per", "its", "their",
}


def split_query(query, sep=METADATA_QUERY_SEPARATOR):
    """Split "<hint>; <question>" on the FIRST separator only (the question
    itself may contain more ';'). Returns (hint, question). If there is no
    separator or either side is empty, returns (None, full_query) so the
    caller can fall back to unfiltered search with the whole query."""
    hint, found, question = query.partition(sep)
    hint, question = hint.strip(), question.strip()
    if not found or not hint or not question:
        return None, query.strip()
    return hint, question


def _tokenize(text):
    """Lowercase word tokens; splits CamelCase ("AcmeHoldingsInc" ->
    acme/holdings/inc) and drops stopwords."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return [t for t in re.findall(r'[a-z0-9]+', text.lower())
            if t not in _METADATA_STOPWORDS]


class _BM25:
    """Minimal Okapi BM25 over a small list of token lists (no extra
    dependency). IDF uses the non-negative Lucene form."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(t) for t in docs_tokens]
        self.dl = np.array([len(t) for t in docs_tokens], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if len(self.dl) and self.dl.mean() > 0 else 1.0
        n = len(docs_tokens)
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query_tokens):
        out = np.zeros(len(self.tfs))
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / norm
        return out


class DocumentMetadataIndex:
    """Stage 1. Document-level 'metadata' = file name + the first
    METADATA_HEADER_CHARS characters (title / parties / preamble for
    contracts, policy name for privacy policies). Built once per run."""

    def __init__(self, documents):
        self.doc_ids = list(documents)
        self.texts, tokens = [], []
        for fp in self.doc_ids:
            name = re.sub(r'[_\-]+', ' ', Path(fp).stem)
            header = documents[fp][:METADATA_HEADER_CHARS]
            self.texts.append(f"{name}\n{header}")
            tokens.append(_tokenize(name) * METADATA_FILENAME_WEIGHT + _tokenize(header))
        self.bm25 = _BM25(tokens)
        self.doc_vecs = None

    def embed(self, provider, model_name):
        """Only needed for the "embedding"/"hybrid" methods."""
        self.doc_vecs = _embed_chunks_raw(
            provider, model_name, [{"text": t} for t in self.texts])

    def select(self, hint, hint_vec=None):
        """Returns (list_of_file_paths, note). `note` is None for a normal
        filtered result, or a short reason when it fell back to ALL documents."""
        all_docs = list(self.doc_ids)
        if hint is None:
            return all_docs, "no '<hint>;' prefix in query -> searched all documents"

        method = METADATA_METHOD
        top = max(1, min(METADATA_TOP_DOCS, len(all_docs)))
        use_bm25 = method in ("bm25", "hybrid")
        use_emb = method in ("embedding", "hybrid")
        if method not in ("bm25", "embedding", "hybrid"):
            raise ValueError(f"Unknown METADATA_METHOD: {method!r}")

        bm = self.bm25.scores(_tokenize(hint)) if use_bm25 else None
        if use_bm25 and bm.max() <= 0:
            if method == "bm25":
                return all_docs, "hint shares no terms with any document -> searched all documents"
            use_bm25 = False  # hybrid degrades to embedding-only

        if method == "bm25":
            order = np.argsort(-bm, kind="stable")
            best = bm[order[0]]
            keep = [i for i in order[:top] if bm[i] >= METADATA_MIN_SCORE_RATIO * best]
            return [self.doc_ids[i] for i in keep], None

        if self.doc_vecs is None or hint_vec is None:
            raise RuntimeError("embedding/hybrid metadata retrieval needs doc and hint vectors")
        fused = np.zeros(len(all_docs))
        for s in ((bm if use_bm25 else None), self.doc_vecs @ hint_vec):
            if s is None:
                continue
            ranks = np.empty(len(s), dtype=int)
            ranks[np.argsort(-s, kind="stable")] = np.arange(len(s))
            fused += 1.0 / (60 + ranks)  # reciprocal-rank fusion
        order = np.argsort(-fused, kind="stable")
        return [self.doc_ids[i] for i in order[:top]], None


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

# -------------------------------- Checks --------------------------------

def spot_check_offsets(chunks_by_doc, documents, sample_size=200, seed=0, show=5):
    """Verify documents[file_path][start:end] == chunk text for a random
    sample of chunks. Returns (pct_correct, failures)."""
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    sample = random.Random(seed).sample(all_chunks, min(sample_size, len(all_chunks)))

    failures, ambiguous = [], 0
    for c in sample:
        doc = documents[c["file_path"]]
        in_bounds = 0 <= c["start"] < c["end"] <= len(doc)
        if not in_bounds or doc[c["start"]:c["end"]] != c["text"]:
            failures.append(c)
        elif doc.count(c["text"]) > 1:
            ambiguous += 1   # text recurs, so the recorded offset could be a duplicate's

    pct = 100 * (len(sample) - len(failures)) / len(sample) if sample else 0.0
    print(f"Offset spot-check: {len(sample) - len(failures)}/{len(sample)} correct ({pct:.1f}%)"
          f"  [{ambiguous} correct but with repeated text]")
    for c in failures[:show]:
        doc = documents[c["file_path"]]
        print(f"  FAIL {c['file_path']} [{c['start']}:{c['end']}]")
        print(f"    chunk : {c['text'][:80]!r}")
        print(f"    source: {doc[c['start']:c['end']][:80]!r}")
    return pct, failures

# -------------------------------- Run --------------------------------

def run():
    if USE_FIXED_DOCUMENT_SUBSET:
        doc_subset = list_first_n_documents(CORPUS_ROOT, FIXED_DOCUMENT_SUBDIR, MAX_DOCUMENTS)
        documents, selected_queries = load_benchmark_fixed_docs(
        BENCHMARK_PATH, CORPUS_ROOT, doc_subset, NUM_QUERIES
    )
    else:
        documents, selected_queries = load_benchmark(
        BENCHMARK_PATH, CORPUS_ROOT, MAX_DOCUMENTS, NUM_QUERIES
    )
    print(f"Loaded {len(documents)} document(s), {len(selected_queries)} quer{'y' if len(selected_queries) == 1 else 'ies'}.")
    for fp in documents:
        print(f"  - {fp}")

    chunk_fn = {"fixed": fixed_chunk, "structure": structure_chunk}[CHUNKING_SCHEME]
    chunks_by_doc = chunk_documents(documents, chunk_fn)
    pct, _ = spot_check_offsets(chunks_by_doc, documents)
    assert pct >= 90, f"Offset spot-check below the 90% success criterion: {pct:.1f}%"
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

    # Stage-1 setup (once per run, shared by all models): split every query
    # into (hint, question) and index the documents' file name + header.
    all_query_texts = [q["query"] for q in selected_queries]
    parsed_queries = [split_query(t) for t in all_query_texts]
    meta_index = DocumentMetadataIndex(documents) if METADATA_RETRIEVAL_ENABLED else None
    if METADATA_RETRIEVAL_ENABLED:
        n_with_hint = sum(1 for h, _ in parsed_queries if h)
        print(f"\nTwo-stage retrieval ON (metadata method: {METADATA_METHOD}, "
              f"top {METADATA_TOP_DOCS} docs). {n_with_hint}/{len(parsed_queries)} queries "
              f"have a '{METADATA_QUERY_SEPARATOR}' hint; the rest search all documents.")
    else:
        print("\nTwo-stage retrieval OFF (full query, whole corpus).")

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
                           "hit_rate": [], "top1_hit_rate": [], "mrr": []}   # added top1_hit_rate
            all_metrics_rerank = {m: [] for m in all_metrics} if RERANK_ENABLED else None
            rerank_failures = 0
            char_totals = {"overlap_chars": 0, "retrieved_chars": 0, "gt_chars": 0}
            char_totals_rerank = dict.fromkeys(char_totals, 0)
            doc_recalls, subset_sizes = [], []   # stage-1 diagnostics

            # Embed every selected query once, up front, in a handful of
            # batched calls (see _embed_queries_raw) rather than one API
            # call per query inside the loop below.
            #
            # query_texts are the full original queries (used for display and
            # the LLM reranker). The vectors used for chunk similarity come
            # from the QUESTION part only when two-stage retrieval is on.
            query_texts = all_query_texts
            sim_texts = ([q for _, q in parsed_queries]
                         if METADATA_RETRIEVAL_ENABLED else query_texts)
            n_batches = (len(sim_texts) + 89) // 90
            print(f"  Embedding {len(sim_texts)} quer{'y' if len(sim_texts) == 1 else 'ies'} "
                  f"in {n_batches} batch call(s)...")
            query_vectors = _embed_queries_raw(provider, model_name, sim_texts)

            # Stage 1: choose the candidate document subset for every query.
            if METADATA_RETRIEVAL_ENABLED:
                hint_vecs = {}
                if METADATA_METHOD in ("embedding", "hybrid"):
                    meta_index.embed(provider, model_name)
                    hint_idx = [i for i, (h, _) in enumerate(parsed_queries) if h]
                    if hint_idx:
                        vecs = _embed_queries_raw(
                            provider, model_name, [parsed_queries[i][0] for i in hint_idx])
                        hint_vecs = dict(zip(hint_idx, vecs))
                subsets = [meta_index.select(h, hint_vecs.get(i))
                           for i, (h, _) in enumerate(parsed_queries)]
            else:
                subsets = [(list(documents), None)] * len(query_texts)

            for qi, q in enumerate(selected_queries):
                gt_spans = q["spans"]
                # Stage 1 result for this query: the candidate document subset.
                subset_docs, subset_note = subsets[qi]
                subset_set = set(subset_docs)
                gt_docs = {s["file_path"] for s in gt_spans}
                missed_docs = gt_docs - subset_set
                doc_recalls.append(1 - len(missed_docs) / len(gt_docs))
                subset_sizes.append(len(subset_docs))
                total_candidates = sum(len(chunks_by_doc[fp]) for fp in subset_docs)

                # Stage 2: plain top-k similarity search via the vector store,
                # scoped to the stage-1 subset, using the query vector embedded
                # above (the question part, when two-stage is on). Asking for
                # all candidates in the subset (not just TOP_K) gives us the
                # full ranking for the diagnostics below at no extra embedding
                # cost -- it's the same query vector, just a bigger k.
                subset_filter = (None if len(subset_docs) == len(documents) else
                                 (lambda doc, _refs=subset_set: doc.metadata["file_path"] in _refs))
                ranked = vector_store.similarity_search_with_score_by_vector(
                    query_vectors[qi].tolist(),
                    k=total_candidates,
                    filter=subset_filter,
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
                for key in char_totals:
                    char_totals[key] += metrics[key]

                # Second stage: hand the top RERANK_CANDIDATE_POOL cosine hits to
                # the LLM, re-sort by its relevance judgment, take the new top-k,
                # and score that the same way. If the LLM call itself fails (bad
                # key, rate limit exhausted, network) we fall back to the cosine
                # metrics for this one query rather than crashing the whole run
                # or silently dropping the query from the reranked average.
                reranked_retrieved, metrics_rerank = None, None
                if RERANK_ENABLED:
                    candidate_pool = full_order_chunks[:RERANK_CANDIDATE_POOL]
                    try:
                        reranked_chunks, _rerank_scores = llm_rerank(query_texts[qi], candidate_pool)
                        reranked_retrieved = reranked_chunks[:TOP_K]
                        metrics_rerank = evaluate_query(reranked_retrieved, gt_spans, k=TOP_K)
                    except Exception as e:
                        rerank_failures += 1
                        print(f"    [rerank failed] {type(e).__name__}: {e} "
                              f"-- using cosine-only metrics for this query's rerank average")
                        reranked_retrieved, metrics_rerank = retrieved, metrics
                    for m in all_metrics_rerank:
                        all_metrics_rerank[m].append(metrics_rerank[m])
                    for key in char_totals_rerank:
                        char_totals_rerank[key] += metrics_rerank[key]

                gt_by_file = _merged_intervals_by_file(gt_spans)

                print(f"\n  Query: {query_texts[qi]}")
                if METADATA_RETRIEVAL_ENABLED:
                    hint_q, question_q = parsed_queries[qi]
                    shown = (subset_docs if len(subset_docs) <= max(METADATA_TOP_DOCS, 1)
                             else f"all {len(subset_docs)} documents")
                    print(f"  Stage 1 (metadata/{METADATA_METHOD}) hint: {hint_q!r} -> "
                          f"{len(subset_docs)}/{len(documents)} doc(s): {shown}")
                    if subset_note:
                        print(f"    [{subset_note}]")
                    if missed_docs:
                        print(f"    [metadata miss] ground-truth doc(s) not in subset: "
                              f"{sorted(missed_docs)}")
                    print(f"  Stage 2 (similarity) question: {question_q!r}")
                print(f"  Ground truth: {[(s['file_path'], s['start'], s['end']) for s in gt_spans]}"
                f" ({metrics['gt_chars']} chars total)")
                for s in gt_spans:
                 print(f"    >>> {documents[s['file_path']][s['start']:s['end']]}")
                if first_hit_rank is None and missed_docs:
                    print(f"  [diagnostic] stage 1 dropped the ground-truth document, so none of "
                          f"the {len(full_order_chunks)} chunks searched could contain the answer "
                          f"(a metadata-retrieval miss, not a chunking/ranking one).")
                elif first_hit_rank is None:
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

                if RERANK_ENABLED and reranked_retrieved is not None and reranked_retrieved is not retrieved:
                    print(f"    -- after LLM rerank (pool of {len(candidate_pool)} cosine candidates) --")
                    for rank, c in enumerate(reranked_retrieved, start=1):
                        c_gt = gt_by_file.get(c["file_path"], [])
                        c_overlap = interval_set_len(intersect_intervals(
                            merge_intervals([(c["start"], c["end"])]), c_gt))
                        preview = c["text"][:ANSWER_PREVIEW_CHARS].replace("\n", " ").strip()
                        print(f"    [{rank}] {c_overlap:4d} chars overlap  "
                              f"{c['file_path']} [{c['start']}:{c['end']}]")
                        print(f"         {preview}...")

                if RERANK_ENABLED:
                    print(f"  Metrics (cosine only):   {metrics}")
                    print(f"  Metrics (+ LLM rerank):  {metrics_rerank}")
                else:
                    print(f"  Metrics: {metrics}")

            avg = {m: round(sum(v) / len(v), 3) for m, v in all_metrics.items()}
            avg.update(micro_prf(**char_totals)) 
            doc_stats = {
                "doc_recall": round(sum(doc_recalls) / len(doc_recalls), 3),
                "avg_docs": round(sum(subset_sizes) / len(subset_sizes), 1),
            }
            print(f"\n  --- {label} (cosine only) averages over {len(all_metrics['hit_rate'])} queries: {avg} ---")
            print(f"  --- stage-1 document recall: {doc_stats['doc_recall']} "
                  f"(avg {doc_stats['avg_docs']} of {len(documents)} docs searched per query) ---")
            summary[label] = {**avg, **doc_stats}

            if RERANK_ENABLED:
                avg_rerank = {m: round(sum(v) / len(v), 3) for m, v in all_metrics_rerank.items()}
                avg_rerank.update(micro_prf(**char_totals_rerank))   
                rerank_label = f"{label} + LLM rerank"
                print(f"  --- {rerank_label} averages over {len(all_metrics_rerank['hit_rate'])} queries: {avg_rerank} ---")
                if rerank_failures:
                    print(f"  [note] {rerank_failures} of {len(selected_queries)} quer"
                          f"{'y' if rerank_failures == 1 else 'ies'} fell back to cosine-only "
                          f"metrics above (the LLM rerank call failed).")
                summary[rerank_label] = {**avg_rerank, **doc_stats}

        except Exception as e:
            # Don't let one provider's failure (bad/expired key, no credits,
            # rate limit, network issue, etc.) take down the whole comparison
            # — report it and move on to the next model.
            print(f"\n  [SKIPPED] {label} failed: {type(e).__name__}: {e}")
            continue

    print(f"\n{'=' * 60}\nSUMMARY (avg over all {len(selected_queries)} queries)\n{'=' * 60}")
    print(f"{'model':56s} {'char_prec@k':>11s} {'char_rec@k':>11s} {'char_f1@k':>10s} "
          f"{'hit_rate':>9s} {'mrr':>6s} {'top1_hit':>9s} {'doc_rec':>8s} {'avg_docs':>9s}"
          f"{'mic_prec':>9s} {'mic_rec':>8s} {'mic_f1':>7s}")
    for label, avg in summary.items():
        print(f"{label:56s} {avg['char_precision@k']:11.3f} {avg['char_recall@k']:11.3f} "
              f"{avg['char_f1@k']:10.3f} {avg['hit_rate']:9.3f} {avg['top1_hit_rate']:9.3f} {avg['mrr']:6.3f} "
              f"{avg['doc_recall']:8.3f} {avg['avg_docs']:9.1f}"
              f"{avg['micro_precision@k']:9.3f} {avg['micro_recall@k']:8.3f} "
              f"{avg['micro_f1@k']:7.3f}")


if __name__ == "__main__":
    run()