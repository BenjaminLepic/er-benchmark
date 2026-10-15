"""
Search module — separated into two independent axes:

  1. Index type (vector store): how vectors are stored and queried
     - sklearn: brute-force cosine via sklearn (works with sparse + dense)
     - faiss_flat: FAISS brute-force inner product (dense only)
     - faiss_hnsw: FAISS HNSW approximate index (dense only)
     - faiss_ivf: FAISS IVF approximate index (dense only)
     - pgvector: Postgres pgvector extension

  2. Retrieval strategy: how candidate pairs are selected from similarities
     - topk: return top-k neighbors per query
     - threshold: return all pairs above similarity threshold
     - topk_threshold: top-k filtered by minimum threshold
     - reciprocal: only keep pairs where both are in each other's top-k
"""

import time
import math
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense(vectors):
    if sparse.issparse(vectors):
        return np.array(vectors.toarray(), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)


def _cross_file_indices(ids, sources, source_a, source_b):
    idx_a = [i for i, s in enumerate(sources) if s == source_a]
    idx_b = [i for i, s in enumerate(sources) if s == source_b]
    return idx_a, idx_b


def automatic_k(n_candidates, multiplier=2.0, maximum=50, minimum=1):
    """Choose a corpus-size-based neighbor pool without dataset-specific tuning."""
    if n_candidates <= 0:
        return 0
    base = math.ceil(math.log2(max(2, n_candidates)))
    return min(n_candidates, maximum, max(minimum, math.ceil(multiplier * base)))


def resolve_retrieval(retrieval, k, n_candidates, params=None):
    """Resolve automatic retrieval aliases to a concrete strategy and k."""
    params = params or {}
    aliases = {
        "auto_topk": "topk",
        "auto_reciprocal": "reciprocal",
    }
    if retrieval not in aliases:
        return retrieval, k
    resolved_k = automatic_k(
        n_candidates,
        multiplier=float(params.get("k_multiplier", 2.0)),
        maximum=int(params.get("max_k", 50)),
        minimum=int(params.get("min_k", 1)),
    )
    return aliases[retrieval], resolved_k


# ---------------------------------------------------------------------------
# Index types — each returns (query_fn, build_time)
#   query_fn(vecs_a, k) -> (scores[n_a, k], indices[n_a, k])
#   query_fn(vecs_a, k=None, threshold=t) -> (scores, indices) per row (ragged)
# We standardize: query_fn returns full sim matrix or top-k results.
# ---------------------------------------------------------------------------

class SklearnIndex:
    """Brute-force cosine via sklearn. Works with sparse and dense."""
    name = "sklearn"

    def __init__(self, vecs_b):
        self.vecs_b = vecs_b

    def topk(self, vecs_a, k, batch_size=200):
        """Returns (scores[n_a, k], indices[n_a, k])."""
        n_a = vecs_a.shape[0]
        k = min(k, self.vecs_b.shape[0])
        all_indices = np.empty((n_a, k), dtype=np.int64)
        all_scores = np.empty((n_a, k), dtype=np.float32)
        for start in range(0, n_a, batch_size):
            end = min(start + batch_size, n_a)
            sims = sk_cosine(vecs_a[start:end], self.vecs_b)
            if k < sims.shape[1]:
                top_idx = np.argpartition(-sims, k, axis=1)[:, :k]
                for i in range(top_idx.shape[0]):
                    order = np.argsort(-sims[i, top_idx[i]])
                    top_idx[i] = top_idx[i][order]
                    all_scores[start + i] = sims[i, top_idx[i]]
            else:
                top_idx = np.argsort(-sims, axis=1)[:, :k]
                for i in range(top_idx.shape[0]):
                    all_scores[start + i] = sims[i, top_idx[i]]
            all_indices[start:end] = top_idx
        return all_scores, all_indices

    def threshold_pairs(self, vecs_a, threshold, batch_size=200):
        """Returns list of (row_idx, col_idx, score) tuples."""
        results = []
        for start in range(0, vecs_a.shape[0], batch_size):
            end = min(start + batch_size, vecs_a.shape[0])
            sims = sk_cosine(vecs_a[start:end], self.vecs_b)
            rows, cols = np.where(sims >= threshold)
            for r, c in zip(rows, cols):
                results.append((start + r, c, sims[r, c]))
        return results


class FaissIndex:
    """Base class for FAISS indexes."""

    def __init__(self, vecs_b, index):
        self.vecs_b = vecs_b
        self.index = index

    def topk(self, vecs_a, k, **kwargs):
        vecs_q = vecs_a.copy()
        import faiss
        faiss.normalize_L2(vecs_q)
        D, I = self.index.search(vecs_q, min(k, self.vecs_b.shape[0]))
        return D, I

    def threshold_pairs(self, vecs_a, threshold, batch_size=200):
        """Fallback: use sklearn for threshold search even with FAISS index."""
        results = []
        for start in range(0, vecs_a.shape[0], batch_size):
            end = min(start + batch_size, vecs_a.shape[0])
            sims = sk_cosine(vecs_a[start:end], self.vecs_b)
            rows, cols = np.where(sims >= threshold)
            for r, c in zip(rows, cols):
                results.append((start + r, c, sims[r, c]))
        return results


class FaissFlatIndex(FaissIndex):
    name = "faiss_flat"

    def __init__(self, vecs_b):
        import faiss
        vecs = vecs_b.copy()
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        super().__init__(vecs_b, index)


class FaissHNSWIndex(FaissIndex):
    name = "faiss_hnsw"

    def __init__(self, vecs_b, M=16, ef_construction=64, ef_search=40):
        import faiss
        vecs = vecs_b.copy()
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        index = faiss.IndexHNSWFlat(dim, M)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(vecs)
        super().__init__(vecs_b, index)


class FaissIVFIndex(FaissIndex):
    name = "faiss_ivf"

    def __init__(self, vecs_b, nlist=100, nprobe=10):
        import faiss
        vecs = vecs_b.copy()
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        actual_nlist = min(nlist, vecs.shape[0])
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist)
        index.train(vecs)
        index.add(vecs)
        index.nprobe = min(nprobe, actual_nlist)
        super().__init__(vecs_b, index)


class PgvectorIndex:
    """Postgres pgvector — uses a temp table with ivfflat or hnsw index."""
    name = "pgvector"

    def __init__(self, vecs_b, ids_b, index_type="hnsw", lists=100, m=16,
                 ef_construction=64, probes=10):
        import psycopg2
        from urllib.parse import urlparse
        import uuid
        db_url = self._env("DATABASE_URL", "")
        if db_url:
            parsed = urlparse(db_url)
            self.conn = psycopg2.connect(
                host=parsed.hostname if parsed.hostname != "db" else "localhost",
                port=parsed.port or 5432,
                dbname=parsed.path.lstrip("/"),
                user=parsed.username,
                password=parsed.password,
            )
        else:
            self.conn = psycopg2.connect(
                host=self._env("PGVECTOR_HOST", "localhost"),
                port=int(self._env("PGVECTOR_PORT", "5432")),
                dbname=self._env("PGVECTOR_DB", "er_benchmark"),
                user=self._env("PGVECTOR_USER", "postgres"),
                password=self._env("PGVECTOR_PASSWORD", ""),
            )
        self.conn.autocommit = True
        self.ids_b = ids_b
        self.dim = vecs_b.shape[1]
        self.index_type = index_type
        self.table = f"_bench_{uuid.uuid4().hex[:8]}"

        cur = self.conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"DROP TABLE IF EXISTS {self.table}")
        cur.execute(f"CREATE TABLE {self.table} (idx int PRIMARY KEY, embedding vector({self.dim}))")

        # Batch insert
        from io import StringIO
        buf = StringIO()
        for i, vec in enumerate(vecs_b):
            vec_str = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
            buf.write(f"{i}\t{vec_str}\n")
        buf.seek(0)
        cur.copy_from(buf, self.table, columns=("idx", "embedding"))

        # Create index
        if index_type == "ivfflat":
            actual_lists = min(lists, vecs_b.shape[0])
            cur.execute(f"""
                CREATE INDEX ON {self.table}
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = {actual_lists})
            """)
            cur.execute(f"SET ivfflat.probes = {min(probes, actual_lists)}")
        elif index_type == "hnsw":
            cur.execute(f"""
                CREATE INDEX ON {self.table}
                USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {ef_construction})
            """)

        cur.close()

    @staticmethod
    def _env(key, default):
        import os
        return os.environ.get(key, default)

    def topk(self, vecs_a, k, **kwargs):
        cur = self.conn.cursor()
        n_a = vecs_a.shape[0]
        all_scores = np.empty((n_a, k), dtype=np.float32)
        all_indices = np.full((n_a, k), -1, dtype=np.int64)

        for i, vec in enumerate(vecs_a):
            vec_str = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
            cur.execute(f"""
                SELECT idx, 1 - (embedding <=> %s::vector) as similarity
                FROM {self.table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (vec_str, vec_str, k))
            for rank, (idx, sim) in enumerate(cur.fetchall()):
                all_indices[i, rank] = idx
                all_scores[i, rank] = sim

        cur.close()
        return all_scores, all_indices

    def threshold_pairs(self, vecs_a, threshold, **kwargs):
        cur = self.conn.cursor()
        results = []
        for i, vec in enumerate(vecs_a):
            vec_str = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
            cur.execute(f"""
                SELECT idx, 1 - (embedding <=> %s::vector) as similarity
                FROM {self.table}
                WHERE 1 - (embedding <=> %s::vector) >= %s
            """, (vec_str, vec_str, threshold))
            for idx, sim in cur.fetchall():
                results.append((i, idx, sim))
        cur.close()
        return results

    def __del__(self):
        try:
            cur = self.conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.close()
            self.conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Index builders — return (index_obj, build_time_s)
# ---------------------------------------------------------------------------

def build_index(index_type, vecs_b, ids_b=None, **params):
    """Build an index, return (index_obj, build_time_seconds)."""
    t0 = time.time()

    if index_type == "sklearn":
        idx = SklearnIndex(vecs_b)
    elif index_type == "faiss_flat":
        idx = FaissFlatIndex(_to_dense(vecs_b))
    elif index_type == "faiss_hnsw":
        idx = FaissHNSWIndex(_to_dense(vecs_b), **{k: v for k, v in params.items()
                             if k in ("M", "ef_construction", "ef_search")})
    elif index_type == "faiss_ivf":
        idx = FaissIVFIndex(_to_dense(vecs_b), **{k: v for k, v in params.items()
                            if k in ("nlist", "nprobe")})
    elif index_type == "pgvector_flat":
        idx = PgvectorIndex(_to_dense(vecs_b), ids_b, index_type="flat")
    elif index_type == "pgvector_hnsw":
        idx = PgvectorIndex(_to_dense(vecs_b), ids_b, index_type="hnsw",
                            **{k: v for k, v in params.items()
                               if k in ("m", "ef_construction")})
    elif index_type == "pgvector_ivfflat":
        idx = PgvectorIndex(_to_dense(vecs_b), ids_b, index_type="ivfflat",
                            **{k: v for k, v in params.items()
                               if k in ("lists", "probes")})
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    build_time = time.time() - t0
    return idx, build_time


# ---------------------------------------------------------------------------
# Retrieval strategies — each takes an index + params, returns candidate pairs
# ---------------------------------------------------------------------------

def retrieve_topk(index, vecs_a, ids, idx_a, idx_b, k=10, **kwargs):
    """Top-k neighbors per query record."""
    t0 = time.time()
    D, I = index.topk(vecs_a, k)
    query_time = time.time() - t0

    pairs = set()
    for i, neighbors in enumerate(I):
        id_a = ids[idx_a[i]]
        for j in neighbors:
            if j >= 0:
                pairs.add((id_a, ids[idx_b[j]]))
    return pairs, query_time


def retrieve_threshold(index, vecs_a, ids, idx_a, idx_b, threshold=0.5, **kwargs):
    """All pairs above similarity threshold."""
    t0 = time.time()
    results = index.threshold_pairs(vecs_a, threshold)
    query_time = time.time() - t0

    pairs = set()
    for r, c, s in results:
        pairs.add((ids[idx_a[r]], ids[idx_b[c]]))
    return pairs, query_time


def retrieve_topk_threshold(index, vecs_a, ids, idx_a, idx_b, k=10, threshold=0.3, **kwargs):
    """Top-k filtered by minimum threshold."""
    t0 = time.time()
    D, I = index.topk(vecs_a, k)
    query_time = time.time() - t0

    pairs = set()
    for i, (neighbors, scores) in enumerate(zip(I, D)):
        id_a = ids[idx_a[i]]
        for j, s in zip(neighbors, scores):
            if j >= 0 and s >= threshold:
                pairs.add((id_a, ids[idx_b[j]]))
    return pairs, query_time


def retrieve_reciprocal(index, vecs_a, vecs_b_for_reverse, ids, idx_a, idx_b, k=10, **kwargs):
    """Keep pair only if mutual top-k."""
    # A -> B
    t0 = time.time()
    _, I_ab = index.topk(vecs_a, k)
    a_to_b = {i: set(ns[ns >= 0].tolist()) for i, ns in enumerate(I_ab)}

    # B -> A: build reverse index
    if sparse.issparse(vecs_a):
        rev_index = SklearnIndex(vecs_a)
    else:
        rev_index = SklearnIndex(np.asarray(vecs_a, dtype=np.float32))
    _, I_ba = rev_index.topk(vecs_b_for_reverse, k)
    b_to_a = {i: set(ns[ns >= 0].tolist()) for i, ns in enumerate(I_ba)}
    query_time = time.time() - t0

    pairs = set()
    for i_a, b_neighbors in a_to_b.items():
        id_a = ids[idx_a[i_a]]
        for j_b in b_neighbors:
            if i_a in b_to_a.get(j_b, set()):
                pairs.add((id_a, ids[idx_b[j_b]]))
    return pairs, query_time


def retrieve_dropoff(index, vecs_a, ids, idx_a, idx_b, k=50, **kwargs):
    """Adaptive cutoff via max similarity gap.

    For each query, fetch top-k neighbors sorted by descending similarity,
    compute gaps between consecutive scores, and cut at the largest gap.
    Keeps everything above the biggest drop-off. Always keeps at least 1.

    The intuition: true matches cluster at high similarity, then there's a
    gap before non-matches. The biggest gap is the natural boundary.
    """
    t0 = time.time()
    D, I = index.topk(vecs_a, k)
    query_time = time.time() - t0

    pairs = set()
    for i in range(D.shape[0]):
        scores = D[i]
        neighbors = I[i]
        # Filter out invalid entries
        valid = [(s, n) for s, n in zip(scores, neighbors) if n >= 0 and s > 0]
        if not valid:
            continue

        id_a = ids[idx_a[i]]
        if len(valid) == 1:
            pairs.add((id_a, ids[idx_b[valid[0][1]]]))
            continue

        # Scores are already sorted descending from topk
        svals = [v[0] for v in valid]

        # Find the largest gap between consecutive similarities
        max_gap = 0
        cut_idx = len(svals)  # default: keep all
        for j in range(len(svals) - 1):
            gap = svals[j] - svals[j + 1]
            if gap > max_gap:
                max_gap = gap
                cut_idx = j + 1  # keep indices 0..j

        # Keep everything above the biggest drop
        for s, n in valid[:cut_idx]:
            pairs.add((id_a, ids[idx_b[n]]))

    return pairs, query_time


def retrieve_knee(index, vecs_a, ids, idx_a, idx_b, k=50, **kwargs):
    """Adaptive cutoff via knee/elbow detection (second derivative).

    Finds where the similarity curve bends most sharply — the point of
    maximum curvature. This is more robust than max-gap when there are
    multiple small drops rather than one clean break.
    """
    t0 = time.time()
    D, I = index.topk(vecs_a, k)
    query_time = time.time() - t0

    pairs = set()
    for i in range(D.shape[0]):
        scores = D[i]
        neighbors = I[i]
        valid = [(s, n) for s, n in zip(scores, neighbors) if n >= 0 and s > 0]
        if not valid:
            continue

        id_a = ids[idx_a[i]]
        if len(valid) <= 2:
            for s, n in valid:
                pairs.add((id_a, ids[idx_b[n]]))
            continue

        svals = np.array([v[0] for v in valid])

        # Second derivative: acceleration of the similarity curve
        # d1[j] = svals[j] - svals[j+1]  (first derivative, positive = dropping)
        # d2[j] = d1[j] - d1[j+1]        (second derivative)
        # Max d2 = where curve transitions from gentle to steep decline
        if len(svals) < 3:
            cut_idx = len(svals)
        else:
            d1 = np.diff(svals)  # first differences (negative since descending)
            d2 = np.diff(d1)     # second differences
            # We want the point where the drop accelerates most
            # d1 is negative, d2 most negative = sharpest acceleration of drop
            knee = np.argmin(d2)  # index into d2
            cut_idx = knee + 2    # +2 because d2 is offset by 2 from original

        for s, n in valid[:cut_idx]:
            pairs.add((id_a, ids[idx_b[n]]))

    return pairs, query_time


def retrieve_stddev(index, vecs_a, ids, idx_a, idx_b, k=50, **kwargs):
    """Adaptive cutoff via standard deviation.

    Keeps neighbors whose similarity is above (mean + 1*std) of the
    top-k scores. The idea: if all top-k are similarly scored, keep few;
    if there's a clear cluster of high scores, keep those.
    Always keeps at least 1.
    """
    t0 = time.time()
    D, I = index.topk(vecs_a, k)
    query_time = time.time() - t0

    pairs = set()
    for i in range(D.shape[0]):
        scores = D[i]
        neighbors = I[i]
        valid = [(s, n) for s, n in zip(scores, neighbors) if n >= 0 and s > 0]
        if not valid:
            continue

        id_a = ids[idx_a[i]]
        svals = np.array([v[0] for v in valid])
        cutoff = svals.mean() + svals.std()

        kept = 0
        for s, n in valid:
            if s >= cutoff:
                pairs.add((id_a, ids[idx_b[n]]))
                kept += 1
        # Always keep at least the best match
        if kept == 0:
            pairs.add((id_a, ids[idx_b[valid[0][1]]]))

    return pairs, query_time


# ---------------------------------------------------------------------------
# Main entry point — used by runner
# ---------------------------------------------------------------------------

def run_search(index_type, retrieval, vectors, ids, sources, source_a, source_b,
               k=10, threshold=0.5, **index_params):
    """
    Run a complete search: build index on B, query with A.

    Returns (candidate_pairs, index_build_time, query_time).
    """
    idx_a, idx_b = _cross_file_indices(ids, sources, source_a, source_b)
    retrieval, k = resolve_retrieval(retrieval, k, len(idx_b), index_params)

    if sparse.issparse(vectors):
        vecs_a = vectors[idx_a]
        vecs_b = vectors[idx_b]
    else:
        vecs_a = np.asarray(vectors[idx_a], dtype=np.float32)
        vecs_b = np.asarray(vectors[idx_b], dtype=np.float32)

    ids_b = [ids[i] for i in idx_b]
    index, build_time = build_index(index_type, vecs_b, ids_b, **index_params)

    dispatch = {
        "topk": lambda: retrieve_topk(index, vecs_a, ids, idx_a, idx_b, k=k),
        "threshold": lambda: retrieve_threshold(index, vecs_a, ids, idx_a, idx_b, threshold=threshold),
        "topk_threshold": lambda: retrieve_topk_threshold(index, vecs_a, ids, idx_a, idx_b, k=k, threshold=threshold),
        "reciprocal": lambda: retrieve_reciprocal(index, vecs_a, vecs_b, ids, idx_a, idx_b, k=k),
        "dropoff": lambda: retrieve_dropoff(index, vecs_a, ids, idx_a, idx_b, k=k),
        "knee": lambda: retrieve_knee(index, vecs_a, ids, idx_a, idx_b, k=k),
        "stddev": lambda: retrieve_stddev(index, vecs_a, ids, idx_a, idx_b, k=k),
    }
    if retrieval not in dispatch:
        raise ValueError(f"Unknown retrieval strategy: {retrieval}")
    pairs, query_time = dispatch[retrieval]()

    return pairs, build_time, query_time


INDEX_TYPES = ["sklearn", "faiss_flat", "faiss_hnsw", "faiss_ivf", "pgvector_flat", "pgvector_hnsw", "pgvector_ivfflat"]
RETRIEVAL_STRATEGIES = [
    "auto_topk", "auto_reciprocal", "topk", "threshold",
    "topk_threshold", "reciprocal", "dropoff", "knee", "stddev",
]
