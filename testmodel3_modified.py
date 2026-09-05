"""
RAG embedding-model comparison, run against a small slice of the real
LegalBench-RAG corpus: OpenAI (text-embedding-3-large) vs Cohere (embed-v4.0)
vs a local, self-hosted model (BAAI/bge-large-en-v1.5). 

SETUP BEFORE RUNNING:
1. Download the corpus + benchmarks from the Dropbox link in the LegalBench-RAG
   README and place them at ./data/corpus/ and ./data/benchmarks/
2. pip install openai cohere sentence-transformers numpy
3. Set your API keys as environment variables:
       export OPENAI_API_KEY="sk-..."
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

NOTE ON COST: OpenAI and Cohere are paid APIs billed per token embedded.
With MAX_DOCUMENTS kept small (1-2) and NUM_QUERIES at 5, this script embeds
a small number of chunks, but every run re-embeds them (no caching), so
increase those numbers gradually rather than jumping to the full corpus.
"""

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() 

import numpy as np

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

TOP_K = 5  # how many chunks to retrieve per query

# Models to compare. "provider" selects which embed_* function is used.
# Swap model names here if you want e.g. text-embedding-3-small or
# embed-english-v3.0 instead.
MODELS = [
    {"provider": "openai", "name": "text-embedding-3-large"},
    {"provider": "cohere", "name": "embed-v4.0"},
    {"provider": "local", "name": "BAAI/bge-large-en-v1.5"},
]

ANSWER_PREVIEW_CHARS = 200  # how much of each retrieved chunk to print
# ======================================================================


# ------------------------------ Chunking ------------------------------

def fixed_chunk(text, chunk_size=FIXED_CHUNK_SIZE, overlap=FIXED_CHUNK_OVERLAP):
    """Split `text` into fixed-size, overlapping windows.
    chunk_size / overlap are character counts, not tokens."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({"text": text[start:end], "start": start, "end": end})
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


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
    if any(lower == term or lower.startswith(term + ' ') for term in heading_terms):
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
    """Return section boundaries using headings first, then paragraph/list boundaries."""
    boundaries = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_no_nl = line.rstrip('\r\n')
        if offset > 0 and _looks_like_heading(line_no_nl):
            boundaries.append(offset)
        offset += len(line)

    # If headings exist, use them directly. Otherwise still preserve paragraph
    # structure instead of treating the entire document as one giant segment.
    if boundaries:
        return sorted(set([0, *boundaries, len(text)]))

    paragraph_boundaries = [0]
    for m in re.finditer(r'\n\s*\n+', text):
        paragraph_boundaries.append(m.end())
    paragraph_boundaries.append(len(text))

    # If paragraph structure is effectively absent, split at sentence/list
    # boundaries only as a last structural fallback.
    if len(paragraph_boundaries) <= 2:
        for m in re.finditer(r'(?<=[.!?])(?:\s+|\n)(?=[A-Z\d])', text):
            paragraph_boundaries.append(m.end())
        paragraph_boundaries = sorted(set([0, *paragraph_boundaries, len(text)]))

    return paragraph_boundaries


def structure_chunk(text, max_size=STRUCTURE_MAX_CHUNK_SIZE, overlap=STRUCTURE_CHUNK_OVERLAP):
    """Chunk by document structure, then split oversized sections with overlap.

    Unlike the old implementation, a missing ``1. DEFINITIONS``-style header
    does not silently convert the whole document into fixed-size windows.
    We first try several heading conventions, then paragraph boundaries, then
    sentence boundaries as a structural fallback.
    """
    boundaries = _structure_boundaries(text)
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
# key just to run the OpenAI + local comparison.

_openai_client = None
_cohere_client = None
_local_models = {}


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set the OPENAI_API_KEY environment variable.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


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


def embed_openai(texts, model_name):
    client = _get_openai_client()
    vectors = []
    for batch in _batched(texts):
        resp = client.embeddings.create(model=model_name, input=batch)
        vectors.extend(d.embedding for d in resp.data)
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


def embed_chunks(provider, model_name, chunks):
    """Embeds chunk text with the given provider/model. Returns an
    (n_chunks, dim) array aligned with `chunks`."""
    texts = [c["text"] for c in chunks]
    if provider == "openai":
        return embed_openai(texts, model_name)
    if provider == "cohere":
        return embed_cohere(texts, model_name, input_type="search_document")
    if provider == "local":
        return embed_local(texts, model_name)
    raise ValueError(f"Unknown provider: {provider}")


def embed_query(provider, model_name, query):
    if provider == "openai":
        return embed_openai([query], model_name)[0]
    if provider == "cohere":
        return embed_cohere([query], model_name, input_type="search_query")[0]
    if provider == "local":
        return embed_local([query], model_name)[0]
    raise ValueError(f"Unknown provider: {provider}")


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

def load_benchmark(benchmark_path, corpus_root, max_documents, num_queries):
    """Walks the benchmark's tests in order, greedily keeping a query only if
    it doesn't push the running set of distinct source documents past
    max_documents. Stops once num_queries queries are collected.

    Adjust the key names below (test['query'], snippet['file_path'],
    snippet['span']) if your downloaded json uses different field names —
    print(raw[0]) after loading to check before running the full thing.
    """
    raw = json.loads(benchmark_path.read_text())
    tests = raw["tests"] if "tests" in raw else raw

    selected_docs = []      # unique file_paths, in first-seen order
    selected_queries = []   # [{"query": str, "spans": [{"file_path","start","end"}]}]

    for test in tests:
        if len(selected_queries) >= num_queries:
            break
        snippet_docs = {s["file_path"] for s in test["snippets"]}
        new_docs = snippet_docs - set(selected_docs)
        if len(selected_docs) + len(new_docs) > max_documents:
            continue  # would need more documents than we're allowed — skip
        selected_docs.extend(sorted(new_docs))
        spans = [{"file_path": s["file_path"], "start": s["span"][0], "end": s["span"][1]}
                 for s in test["snippets"]]
        selected_queries.append({"query": test["query"], "spans": spans})

    documents = {}
    for file_path in selected_docs:
        full_path = corpus_root / file_path
        if full_path.exists():
            documents[file_path] = full_path.read_text()
        else:
            print(f"  [warn] corpus file not found, skipping: {full_path}")

    return documents, selected_queries


def _chunks_overlap(a, b):
    """True when two chunks from the same file overlap in character offsets."""
    return (
        a["file_path"] == b["file_path"]
        and a["start"] < b["end"]
        and b["start"] < a["end"]
    )


def select_top_k_with_overlap(chunks, scores, ranked_indices, k):
    """Select at most k chunks while keeping overlapping neighbors together.

    Retrieval starts from the best semantic hit. Once a chunk is selected, its
    highest-scoring overlapping neighbors are preferred for the remaining slots.
    Only after the available overlap is exhausted do we fill from the remaining
    semantic ranking. Thus overlap is useful *inside* the real TOP_K budget.

    No extra chunks are appended after TOP_K, and the final list remains ordered
    by the original cosine-similarity score so MRR/rank metrics retain their usual
    interpretation.
    """
    if k <= 0 or len(ranked_indices) == 0:
        return []

    k = min(k, len(ranked_indices))
    ranked = [int(i) for i in ranked_indices]
    rank_position = {idx: rank for rank, idx in enumerate(ranked)}

    selected = [ranked[0]]
    selected_set = {ranked[0]}

    while len(selected) < k:
        overlap_candidates = [
            idx for idx in ranked
            if idx not in selected_set
            and any(_chunks_overlap(chunks[idx], chunks[s]) for s in selected)
        ]

        if overlap_candidates:
            # Among overlapping neighbors, preserve semantic quality.
            # rank_position breaks exact-score ties deterministically.
            best_idx = max(
                overlap_candidates,
                key=lambda i: (float(scores[i]), -rank_position[i]),
            )
        else:
            # No overlap remains: return to ordinary semantic ranking.
            best_idx = next(i for i in ranked if i not in selected_set)

        selected.append(best_idx)
        selected_set.add(best_idx)

    # Keep standard descending-score order for downstream rank metrics.
    selected.sort(key=lambda i: (-float(scores[i]), rank_position[i]))
    return selected


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
            struct_boundaries = _structure_boundaries(documents[fp])
            n_boundaries = max(0, len(struct_boundaries) - 2)
            print(f"    {n_boundaries} structural boundaries detected")
            if n_boundaries == 0:
                print("    [info] No standalone headings detected; using paragraph/sentence "
                      "boundaries rather than silently treating the whole document as one section.")

    summary = {}

    for model_cfg in MODELS:
        provider, model_name = model_cfg["provider"], model_cfg["name"]
        label = f"{provider}/{model_name}"
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

        try:
            embeddings_by_doc = {
                fp: embed_chunks(provider, model_name, chunks)
                for fp, chunks in chunks_by_doc.items()
            }

            all_metrics = {"char_precision@k": [], "char_recall@k": [], "char_f1@k": [],
                           "hit_rate": [], "mrr": []}

            for q in selected_queries:
                query_text = q["query"]
                gt_spans = q["spans"]
                ref_docs = sorted({s["file_path"] for s in gt_spans})

                combined_chunks = [c for fp in ref_docs for c in chunks_by_doc[fp]]
                combined_embs = np.vstack([embeddings_by_doc[fp] for fp in ref_docs])

                query_emb = embed_query(provider, model_name, query_text)
                scores = combined_embs @ query_emb
                full_order = np.argsort(scores)[::-1]

                # Re-rank within the TOP_K budget so overlap is actually useful.
                # Start from semantic ranking, then preferentially retain chunks
                # that overlap a selected chunk. This prevents an overlapping
                # neighbor from being retrieved only after TOP_K has already
                # been filled with unrelated chunks. The total returned count
                # never exceeds TOP_K.
                top_idx = select_top_k_with_overlap(
                    combined_chunks, scores, full_order, TOP_K
                )
                retrieved = [combined_chunks[i] for i in top_idx]

                # Diagnostic: rank of the first chunk (in the FULL ranking, not
                # just top-k) that overlaps the ground truth at all. Distinguishes
                # "the answer chunk exists but ranked below top-k" (a retrieval/
                # ranking issue) from "no chunk in the pool overlaps the ground
                # truth at all" (a chunking issue — the answer got cut apart or
                # was never captured as a coherent chunk to begin with).
                gt_by_file_full = _merged_intervals_by_file(gt_spans)
                first_hit_rank = None
                for rank, i in enumerate(full_order, start=1):
                    c = combined_chunks[i]
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
                selected_set = set(top_idx)
                if first_hit_rank is None:
                    print(f"  [diagnostic] NO chunk among all {len(combined_chunks)} candidate "
                          f"chunks overlaps the ground truth at all — the answer text isn't "
                          f"captured by any generated chunk.")
                elif not any(i in selected_set for i in range(len(combined_chunks))
                             if interval_set_len(intersect_intervals(
                                 merge_intervals([(combined_chunks[i]["start"], combined_chunks[i]["end"])]),
                                 gt_by_file_full.get(combined_chunks[i]["file_path"], [])
                             )) > 0):
                    print(f"  [diagnostic] first semantic hit was rank #{first_hit_rank} of "
                          f"{len(combined_chunks)}, but no overlapping chunk survived the "
                          f"overlap-aware top_k={TOP_K} selection.")
                elif first_hit_rank > TOP_K:
                    print(f"  [diagnostic] first semantic hit ranked #{first_hit_rank} of "
                          f"{len(combined_chunks)}; overlap-aware top_k={TOP_K} may still "
                          f"select a neighboring chunk if it overlaps a stronger hit.")
                for rank, (i, c) in enumerate(zip(top_idx, retrieved), start=1):
                    c_gt = gt_by_file.get(c["file_path"], [])
                    c_overlap = interval_set_len(intersect_intervals(
                        merge_intervals([(c["start"], c["end"])]), c_gt))
                    preview = c["text"][:ANSWER_PREVIEW_CHARS].replace("\n", " ").strip()
                    print(f"    [{rank}] {c_overlap:4d} chars overlap  "
                          f"{c['file_path']} [{c['start']}:{c['end']}] score={scores[i]:.3f}")
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