# LLM-Based Hybrid Recommendation System

Recommender systems fail hardest on the users they know least about. This work
attacks that directly: a hybrid architecture that fuses classical collaborative
filtering with a fine-tuned SBERT two-tower model, so that when interaction
history runs out, semantic understanding of the items takes over.

On cold-start users — people the model has never seen before — it improves
**NDCG@5 by 32.4%** over a collaborative filtering baseline.

> Undergraduate thesis, Bahá'í Institute for Higher Education, June 2025.
> Supervised by Dr. Fares Hedayati and Dr. Holakou Rahmanian.

## Results

### Overall

| | NDCG@5 | NDCG@10 |
|---|---|---|
| Improvement of hybrid model over baseline | **+5.8%** | **+8.94%** |

### Cold-start users

The harder and more interesting case — users with no interaction history in
training.

| | NDCG@5 | NDCG@10 |
|---|---|---|
| Improvement over collaborative filtering | **+32.44%** | **+16.51%** |

### Model comparison — MovieLens 1M

| Model | NDCG@5 | NDCG@10 | Recall@5 | MRR@5 |
|---|---|---|---|---|
| **Hybrid (RRF)** | 0.7103 | **0.5970** | 0.0253 | 0.1285 |
| SBERT two-tower | **0.7358** | 0.5558 | 0.0032 | 0.0270 |
| Collaborative filtering | 0.6714 | 0.5480 | **0.0300** | **0.1339** |

The interesting result is not that one model won. **The SBERT two-tower model
has the best NDCG@5 but collapses on Recall and MRR; collaborative filtering is
the reverse.** Each is strong exactly where the other is weak. The hybrid, fused
by reciprocal rank fusion, is the only configuration that holds up across all
metrics at once — second-best NDCG@5, best NDCG@10, and Recall and MRR close to
collaborative filtering's.

A single-metric evaluation would have picked the wrong model.

## Approach

Two towers, one per side of the recommendation problem, then a fusion step.

```
   User text                          Item text
(history, tags, genres)         (title, metadata, genres)
        │                                  │
        ▼                                  ▼
   SBERT encoder  ◄── fine-tuned ──►  SBERT encoder
        │                                  │
        └────────► similarity ◄────────────┘
                       │
                       ▼
        Reciprocal Rank Fusion  ◄── Collaborative filtering
             (weights 4 : 1)              (SVD / user- / item-based)
                       │
                       ▼
                Ranked recommendations
```

The 4:1 weighting favouring the SBERT model was selected empirically as the best
of the configurations tested.

**Backbone** — `paraphrase-MiniLM-L12-v2`, the most effective of the SBERT
variants evaluated.

## Ablation studies

Eight experiments isolating what actually drives ranking quality. Half of them
failed, which is the useful half.

| Experiment | Result |
|---|---|
| Tags and tokens in the item text | **Helped** — meaningfully |
| User history in the user text | **Helped** — the largest single gain |
| Binarizing ratings | Did not help |
| Including negative user history | Did not improve performance |
| Weighting parts of the text | Did not work as expected |
| Adding layers to the tower | **Hurt** — opposite of the intended effect |
| Training on genres alone | Surprising behaviour, worth further study |
| SBERT backbone selection | `paraphrase-MiniLM-L12-v2` best of those tested |

## Datasets

- **MovieLens 1M** — one million ratings, the standard recommender benchmark
- **Amazon Books** — a larger, sparser, more realistic catalogue

Cold-start evaluation was constructed by splitting interactions so that 10% of
test users appear nowhere in the training data.

## Metrics

NDCG@5 and NDCG@10 (three variants), Recall@5/@10, MRR@5/@10. NDCG is the
primary metric — it rewards putting relevant items high in the ranking, which is
what actually matters to a user looking at the first screen of results.

## Baselines

- SVD-based collaborative filtering (`CF_SVD.py`)
- User-based collaborative filtering (`CF_user_based.py`)
- Item-based collaborative filtering (`CF_item_based.py`)
- BERT4Rec
- SBERT two-tower alone

## Repository layout

```
TwoTowerModel.py          two-tower architecture
CF_SVD.py                 SVD collaborative filtering baseline
CF_user_based.py          user-based CF baseline
CF_item_based.py          item-based CF baseline
MovieLens Codes/          MovieLens 1M experiments
Amazon Beauty Codes/      Amazon experiments
dataset/ml-1m/            MovieLens data
log_tensorboard/          training logs
saved/                    model checkpoints
```

## Stack

Python · PyTorch Lightning · CUDA · SBERT (Sentence-Transformers) ·
scikit-learn · pandas · NumPy · TensorBoard

## Running it

```bash
pip install -r requirements.txt    # [FILL: add a requirements.txt]
```

**[FILL: which notebook or script reproduces the headline result, and where the
data should sit. Two or three lines — a reader who can't run it will not try
twice.]**

## Limitations and future work

**Transformer layers for embedding generation.** Only a limited set of model
configurations was explored. Self-attention should in principle identify which
parts of a user's history and an item's description carry the most signal, and
produce better embeddings as a result. The transformer variants tested here did
not improve on the SBERT baseline — but the search was shallow, and this is the
most likely route to fixing the two-tower model's weak Recall and MRR.

**Comparison against a RAG-based recommender.** A prompt-based LLM using
retrieval-augmented generation, including few-shot variants, would be the
natural point of comparison for this architecture. If it performed well it could
be folded into the fusion step as a third signal.

**Balancing the fusion.** The hybrid improves NDCG substantially while trailing
collaborative filtering slightly on Recall and MRR. Closing that remaining gap
without giving up ranking quality is the clearest open problem here.

## Full thesis

*Leveraging Large Language Models in Hybrid Recommendation Systems*
Hooman Katebpour Shahidi, June 2025.
Supervised by Dr. Fares Hedayati and Dr. Holakou Rahmanian.

**[FILL: commit the PDF to this repo and link it here — simplest option — or
link the arXiv preprint once it is up.]**

---

### Housekeeping before publishing

`log/`, `log_tensorboard/` and `saved/` are training artefacts and model
checkpoints tracked in git. Once you've read the numbers off TensorBoard:

```bash
git rm -r --cached log log_tensorboard saved
printf 'log/\nlog_tensorboard/\nsaved/\n' >> .gitignore
git commit -m "Stop tracking training artefacts"
git push
```
