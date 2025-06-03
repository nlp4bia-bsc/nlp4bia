**BECELinker: A Two‐Stage “Retrieve & Rerank” Entity Linker**

BECELinker means **B**i-**E**ncoder + **C**ross-**E**ncoder Linker. It’s a Python module designed to efficiently link mentions in text to entries in a large gazetteer (like a medical concept database) using a two-stage approach: first, fast retrieval with a bi-encoder, then fine-grained reranking with a cross-encoder.

It avoids using libraries with difficult dependencies (like `faiss`) and instead relies on the `sentence-transformers` library for embeddings and tensor operations using `torch` and `numpy`. This makes it easy to integrate into existing NLP pipelines without needing complex setups.

---
- [1. Why Two Stages?](#1-why-two-stages)
- [2. What BECELinker Does](#2-what-becelinker-does)
- [3. Theoretical Background (In Plain English)](#3-theoretical-background-in-plain-english)
- [4. Quick Example of Use](#4-quick-example-of-use)
- [5. When to Use BECELinker](#5-when-to-use-becelinker)
- [6. Tips \& Tricks](#6-tips--tricks)
- [7. In a Nutshell](#7-in-a-nutshell)

### 1. Why Two Stages?

1. **Bi-Encoder (Fast Retrieval)**

   * Think of each text snippet (a “mention”) and each gazetteer entry (“candidate term”) being separately mapped into the same vector space.
   * For instance, “aspirin” → a 768-dim embedding; “C001 | aspirin” in your gazetteer → another 768-dim embedding.
   * At runtime, you encode a mention, then quickly find the top-K nearest gazetteer entries by cosine similarity. This is extremely fast—even if you have hundreds of thousands of terms—because it’s just a matrix lookup or approximate nearest-neighbor search.

2. **Cross-Encoder (Fine-Grained Reranking)**

   * Once you have, say, the top 50 candidates from the bi-encoder, you want to score them more precisely. A Cross-Encoder takes a pair `(mention, candidate_term)` and runs both texts through a single transformer, letting it attend across the two.
   * This yields a high-quality relevance score (e.g., “How likely is it that this mention refers to that exact gazetteer entry?”). Because you only apply it to 50 pairs instead of 50 × 100,000, it’s still practical on GPU.

By combining both, you get the **speed** of a bi-encoder (to narrow down millions of possibilities) and the **accuracy** of a cross-encoder (to pick the best from those few candidates).

---

### 2. What BECELinker Does

1. **Builds or Accepts a Precomputed “Vector DB”**

   * You give it a pandas DataFrame of gazetteer entries with two columns:

     * `term` (the text string of each entry)
     * `code` (the unique identifier for that entry)
   * It either (a) encodes all those `term` strings into embeddings using a SentenceTransformer, or (b) takes a tensor you’ve already computed.

2. **Retrieves Top-K Candidates (Bi-Encoder Stage)**

   * Call `.retrieve_top_k(mentions, k=…)`.
   * If you pass a list of raw mention strings, it encodes them, computes cosine similarity against every gazetteer embedding, and returns, for each mention, the top K codes/terms + their bi-encoder scores.

3. **Reranks with a Cross-Encoder**

   * You feed those (mention → candidate list) results into the cross-encoder reranker.
   * It builds all `(mention, term)` pairs, scores each pair with CrossEncoder.predict(…), then sorts each mention’s candidates by that new score.
   * You end up with a final list of codes/terms ordered by cross-encoder confidence.

4. **Returns Final, Sorted Lists**

   * Each mention yields a dictionary containing:

     * (optionally) `"mention"`: the original string
     * `"terms"`: the candidate terms sorted by cross-encoder score
     * `"codes"`: their corresponding unique IDs
     * `"similarity"`: the cross-encoder scores (higher = more confident)

---

### 3. Theoretical Background (In Plain English)

* **Why not just use a Cross-Encoder on all candidates?**
  A Cross-Encoder needs to run a transformer on each pair—in practice, if you have 100,000 possible entries, scoring all 100,000 pairs per mention is prohibitively slow (each transformer pass can take 0.01–0.05 s).
* **Why not only a Bi-Encoder?**
  A bi-encoder encodes texts independently. It can miss subtleties: e.g. “cold” might map near “chicken soup” in embedding space, but the Cross-Encoder could catch that “cold” here refers to “C0027 | common cold,” not the soup recipe.
* **Retrieve & Rerank = Best of Both**
  First, we quickly narrow from 100,000 → 50 using the bi-encoder. Then we spend a bit more compute to sort those 50 precisely with the cross-encoder. This two-step avoids huge costs while keeping high accuracy.

---

### 4. Quick Example of Use

```python
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from nlp4bia.linking.BECELinker import BECELinker

# 1) Load your gazetteer as a DataFrame with "term" and "code" columns:
gaz_proc = pd.read_csv("medproc_gazetteer.csv")  # must have columns ["term","code"]

# You can also use one of the preprocessed gazetteers from nlp4bia:
# from nlp4bia.datasets.benchmark.medprocner import MedprocnerLoader, MedprocnerGazetteer
# gaz_proc = MedprocnerLoader().df

# 2) Prepare or load your bi-encoder & cross-encoder:
biencoder_path = "/path/to/bi_encoder_checkpoint"  # or a preloaded SentenceTransformer object
crossencoder_path = "/path/to/cross_encoder_checkpoint"  # or a preloaded CrossEncoder object

# 3) Instantiate BECELinker:
linker = BECELinker(
    df_gazetteer=gaz_proc,
    biencoder_model_or_path=biencoder_path,           # can be a model instance or a path string
    crossencoder_model_or_path=crossencoder_path,     # can be a model instance or path string
    n_candidates=50,                     # how many candidates to keep after bi-encoder retrieval
    top_k=100,                           # how many to retrieve before reranking
    normalize_embeddings=True,
    show_progress_bar=True
)

# 4) Prepare a list of mention strings you want to link:
mentions = ["aspirin", "heart attack", "pulmonary embolism"]

# 5) Link them:
results = linker.link(mentions, top_k=100, return_documents=True)

# 6) Inspect results for "aspirin":
res0 = results[0]
print("Mention:", res0["mention"])
for rank, (term, code, score) in enumerate(zip(res0["terms"], res0["codes"], res0["similarity"]), start=1):
    print(f" {rank:02d}. {term} (ID: {code}) → score: {score:.4f}")
```

**What happens under the hood?**

1. **Bi-Encoder Pass**

   * `"aspirin"` → a 768-dim float vector.
   * Compute cosine‐similarity vs. every gazetteer‐term vector → pick the top 100 IDs & terms.

2. **Cross-Encoder Pass**

   * Build 100 pairs:

     ```
     [("aspirin","aspirin"), ("aspirin","acetylsalicylic acid"), …]
     ```
   * Each pair goes through the Cross-Encoder, which outputs a single “matching score.”
   * We sort those 100 by that score; the highest becomes our final “best candidate.”

3. **Return Sorted List**

   * You get back something like:

     ```
     {
       "mention": "aspirin",
       "terms": ["aspirin", "acetylsalicylic acid", "ibuprofen", …],
       "codes": ["C001241",        "C003184",             "C0020746", …],
       "similarity": [5.12,          3.98,                  2.47, …]
     }
     ```

---

### 5. When to Use BECELinker

* **Large Gazetteer (Millions of Entries):**  You cannot brute-force score every mention against every entry with a Cross-Encoder.
* **Need Both Speed & Accuracy:**  A standalone bi-encoder is fast but can be too crude; a standalone cross-encoder is accurate but too slow if used on the full set.
* **Plug & Play:**  Swap in any SentenceTransformer bi-encoder or HuggingFace Cross-Encoder checkpoint without rewriting retrieval code.

---

### 6. Tips & Tricks

* **Choose `top_k` Wisely**:  If you set `top_k` = 100, you’ll spend \~100 GPU passes per mention in stage 2. If that’s too slow, reduce to 50 or 20—but doing so risks dropping the correct candidate from stage 1.
* **Normalization Matters**:  If your bi-encoder was trained with cosine-based loss, pass `normalize_embeddings=True`. Otherwise, results may be suboptimal.
* **Batch Sizes**:  If you run out of GPU memory during the cross-encoder step, lower the `reranker_batch_size` inside BECELinker’s constructor.

---

### 7. In a Nutshell

* **BECELinker** = “**B**i-encoder **E**mbedding retrieval” + “**C**ross-Encoder **E**Xcellence reranking.”
* It ensures you can handle very large gazetteers (e.g., clinical concept databases, product catalogs, knowledge bases) in a practical way—fast enough to deploy, accurate enough for real‐world needs.
* Just point it at your gazetteer, load your pre‐trained models, hand it a list of mentions, and get back neatly sorted candidate lists (codes + terms + scores).

With BECELinker, you get both efficiency and precision in one easy-to-use wrapper.
