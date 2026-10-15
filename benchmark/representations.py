import hashlib
import os
import shutil

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "embeddings")


def _embedding_cache_path(namespace, texts):
    digest = hashlib.sha256(namespace.encode("utf-8"))
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return os.path.join(_CACHE_DIR, f"{digest.hexdigest()}.npz")


def _load_embedding_cache(namespace, texts):
    path = _embedding_cache_path(namespace, texts)
    if not os.path.exists(path):
        return None, path
    with np.load(path) as cached:
        vectors = np.asarray(cached["vectors"], dtype=np.float32)
    if vectors.shape[0] != len(texts):
        return None, path
    return vectors, path


def _save_embedding_cache(path, vectors):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.npz"
    np.savez(temporary, vectors=np.asarray(vectors, dtype=np.float32))
    os.replace(temporary, path)


def _adaptive_feature_cap(texts):
    """Scale vocabulary capacity with corpus size, within practical bounds."""
    return min(100_000, max(10_000, len(texts) * 10))


def tfidf(texts, **kwargs):
    """TF-IDF representation. Returns sparse matrix."""
    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    return vec.fit_transform(texts), "tfidf"


def tfidf_adaptive(texts, **kwargs):
    """Word TF-IDF with a dataset-size-derived vocabulary cap."""
    cap = _adaptive_feature_cap(texts)
    vec = TfidfVectorizer(max_features=cap, sublinear_tf=True)
    return vec.fit_transform(texts), f"tfidf-adaptive-{cap}"


def bm25(texts, **kwargs):
    """BM25 representation as sparse matrix (via TF-IDF with BM25-style weighting)."""
    return _bm25(texts, max_features=5000), "bm25"


def bm25_adaptive(texts, **kwargs):
    """BM25 with a dataset-size-derived vocabulary cap."""
    cap = _adaptive_feature_cap(texts)
    return _bm25(texts, max_features=cap), f"bm25-adaptive-{cap}"


def _bm25(texts, max_features):
    from sklearn.feature_extraction.text import CountVectorizer
    from scipy import sparse

    cv = CountVectorizer(max_features=max_features)
    tf_raw = cv.fit_transform(texts).astype(np.float64)

    # BM25 parameters
    k1 = 1.5
    b = 0.75
    n_docs = tf_raw.shape[0]

    # Document lengths and average
    doc_lens = tf_raw.sum(axis=1).A1  # dense array of doc lengths
    avgdl = doc_lens.mean()

    # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    df = (tf_raw > 0).sum(axis=0).A1
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    # BM25 TF: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    # Process row-by-row to keep sparse
    rows, cols, vals = [], [], []
    csc = tf_raw.tocsc()
    for j in range(tf_raw.shape[1]):
        col = csc.getcol(j)
        for idx in range(col.nnz):
            i = col.indices[idx]
            tf_val = col.data[idx]
            norm_tf = tf_val * (k1 + 1) / (tf_val + k1 * (1 - b + b * doc_lens[i] / avgdl))
            score = norm_tf * idf[j]
            if score > 0:
                rows.append(i)
                cols.append(j)
                vals.append(score)

    bm25_mat = sparse.csr_matrix((vals, (rows, cols)), shape=tf_raw.shape)
    # L2 normalize rows for cosine similarity
    from sklearn.preprocessing import normalize
    bm25_mat = normalize(bm25_mat, norm='l2')
    return bm25_mat


def trigram(texts, **kwargs):
    """Character trigram TF-IDF representation. Returns sparse matrix."""
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), max_features=5000, sublinear_tf=True)
    return vec.fit_transform(texts), "trigram"


def trigram_adaptive(texts, **kwargs):
    """Character 3-5 gram TF-IDF with a dataset-size-derived vocabulary cap."""
    cap = _adaptive_feature_cap(texts)
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=cap,
        sublinear_tf=True,
    )
    return vec.fit_transform(texts), f"trigram-adaptive-{cap}"


def hybrid_adaptive(texts, **kwargs):
    """Equal-weight late-vector fusion of word and character TF-IDF."""
    from scipy import sparse
    from sklearn.preprocessing import normalize

    cap = _adaptive_feature_cap(texts)
    word = TfidfVectorizer(max_features=cap, sublinear_tf=True).fit_transform(texts)
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=cap,
        sublinear_tf=True,
    ).fit_transform(texts)
    vectors = sparse.hstack((word, char), format="csr")
    vectors = normalize(vectors, norm="l2")
    return vectors, f"hybrid-adaptive-{cap}x2"


def sbert_minilm(texts, **kwargs):
    """SBERT all-MiniLM-L6-v2 (384 dims)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=128)
    return np.array(vecs, dtype=np.float32), "sbert-minilm-384"


def sbert_mpnet(texts, **kwargs):
    """SBERT all-mpnet-base-v2 (768 dims)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    vecs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=64)
    return np.array(vecs, dtype=np.float32), "sbert-mpnet-768"


def openai_small_256(texts, **kwargs):
    """OpenAI text-embedding-3-small at 256 dims."""
    return _openai_embed(texts, "text-embedding-3-small", 256), "openai-small-256"


def openai_small_512(texts, **kwargs):
    """OpenAI text-embedding-3-small at 512 dims."""
    return _openai_embed(texts, "text-embedding-3-small", 512), "openai-small-512"


def openai_small_1536(texts, **kwargs):
    """OpenAI text-embedding-3-small at 1536 dims."""
    return _openai_embed(texts, "text-embedding-3-small", 1536), "openai-small-1536"


def openai_large_3072(texts, **kwargs):
    """OpenAI text-embedding-3-large at 3072 dims."""
    return _openai_embed(texts, "text-embedding-3-large", 3072), "openai-large-3072"


def _estimate_tokens(text):
    return len(text) // 3


def _build_token_aware_batches(texts, max_tokens=250_000, max_items=2048):
    batches = []
    current_batch = []
    current_tokens = 0
    for text in texts:
        est = _estimate_tokens(text)
        if current_batch and (current_tokens + est > max_tokens or len(current_batch) >= max_items):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(text)
        current_tokens += est
    if current_batch:
        batches.append(current_batch)
    return batches


def _openai_embed(texts, model, dimensions):
    """Batch-embed via OpenAI API with token-aware batching, returns L2-normalised numpy array."""
    import openai

    namespace = f"openai:{model}:dimensions={dimensions}"
    cached, cache_path = _load_embedding_cache(namespace, texts)
    if cached is not None:
        return cached

    client = openai.OpenAI()
    all_vecs = []
    batches = _build_token_aware_batches(texts)
    for batch in batches:
        resp = client.embeddings.create(input=batch, model=model, dimensions=dimensions)
        vecs = [e.embedding for e in resp.data]
        all_vecs.extend(vecs)
    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    arr = arr / norms
    _save_embedding_cache(cache_path, arr)
    return arr


def gemini_768(texts, **kwargs):
    """Gemini gemini-embedding-001 at 768 dims (RETRIEVAL_QUERY)."""
    return _gemini_embed(texts, 768, task_type="RETRIEVAL_QUERY"), "gemini-768"


def gemini_1536(texts, **kwargs):
    """Gemini gemini-embedding-001 at 1536 dims (RETRIEVAL_QUERY)."""
    return _gemini_embed(texts, 1536, task_type="RETRIEVAL_QUERY"), "gemini-1536"


def gemini_3072(texts, **kwargs):
    """Gemini gemini-embedding-001 at 3072 dims (RETRIEVAL_QUERY)."""
    return _gemini_embed(texts, 3072, task_type="RETRIEVAL_QUERY"), "gemini-3072"


def _gemini_embed(texts, dimensions, task_type="RETRIEVAL_QUERY", batch_size=100):
    """Batch-embed via Google Gemini API, returns L2-normalised numpy array.
    Free tier: 100 req/min, so we throttle and retry on 429."""
    from google import genai
    import time as _time

    namespace = f"google:gemini-embedding-001:dimensions={dimensions}:task={task_type}"
    cached, cache_path = _load_embedding_cache(namespace, texts)
    if cached is not None:
        return cached

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    parts_dir = f"{cache_path}.parts"
    os.makedirs(parts_dir, exist_ok=True)
    part_paths = sorted(
        os.path.join(parts_dir, filename)
        for filename in os.listdir(parts_dir)
        if filename.endswith(".npy")
    )
    part_arrays = [np.load(path) for path in part_paths]
    processed = sum(part.shape[0] for part in part_arrays)
    if any(part.ndim != 2 or part.shape[1] != dimensions for part in part_arrays):
        shutil.rmtree(parts_dir)
        os.makedirs(parts_dir, exist_ok=True)
        part_arrays = []
        processed = 0

    for i in range(processed, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(8):
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config={
                        "output_dimensionality": dimensions,
                        "task_type": task_type,
                    },
                )
                batch_array = np.asarray(
                    [emb.values for emb in result.embeddings],
                    dtype=np.float32,
                )
                part_path = os.path.join(parts_dir, f"{i:09d}.npy")
                np.save(part_path, batch_array)
                part_arrays.append(batch_array)
                break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "503" in err_str) and attempt < 7:
                    wait = min(2 ** attempt + 5, 65)
                    print(f"Gemini {err_str[:30]}... waiting {wait}s (batch {i//batch_size + 1})")
                    _time.sleep(wait)
                else:
                    raise
    arr = np.concatenate(part_arrays, axis=0)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    arr = arr / norms
    _save_embedding_cache(cache_path, arr)
    shutil.rmtree(parts_dir)
    return arr


REPRESENTATIONS = {
    "tfidf": tfidf,
    "tfidf-adaptive": tfidf_adaptive,
    "bm25": bm25,
    "bm25-adaptive": bm25_adaptive,
    "trigram": trigram,
    "trigram-adaptive": trigram_adaptive,
    "hybrid-adaptive": hybrid_adaptive,
    "sbert-minilm": sbert_minilm,
    "sbert-mpnet": sbert_mpnet,
    "openai-small-256": openai_small_256,
    "openai-small-512": openai_small_512,
    "openai-small-1536": openai_small_1536,
    "openai-large-3072": openai_large_3072,
    "gemini-768": gemini_768,
    "gemini-1536": gemini_1536,
    "gemini-3072": gemini_3072,
}


SPARSE_REPRESENTATIONS = {
    "tfidf",
    "tfidf-adaptive",
    "bm25",
    "bm25-adaptive",
    "trigram",
    "trigram-adaptive",
    "hybrid-adaptive",
}
