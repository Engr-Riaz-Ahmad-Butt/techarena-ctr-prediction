# nx_research.md - signal families from comparable competitions (web research, 2026-10-29)

Scope: next-click / session / slate competitions and sequential-rec literature. Everything in the
CLOSED list (slate composition, directed last->candidate transitions, ItemKNN co-click, repeat/negative-
exposure, short-history specialists, reweighting, stackers, seeds, EASE, residual DIN/GRU/attention,
timing/session, weekday, XGB merges, id categoricals, tied-item mean-pooled factors, exposure-aware ALS,
pairwise/listwise residuals, NCE/Bernoulli factor objectives) is excluded or only referenced as a boundary.
Cost column assumes CPU, 2-5 GB RAM, ~686k training events (6.9M candidate rows), 50k items, 53k users.
Caveat: Kaggle discussion pages do not render for fetch tools, so OTTO/H&M details come from mirrored
summaries and public repos; those are flagged "(secondary)".

## Part 1 - Signal families not yet tried

### F1. Time- and position-decayed, multi-anchor, popularity-normalised covisitation scores
- Source: OTTO 2023 (1st/2nd/3rd places, secondary: https://lukan217.github.io/2023/02/12/Kaggle%20Otto%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E6%AF%94%E8%B5%9BTOP%E6%96%B9%E6%A1%88%E6%80%BB%E7%BB%93/ ; 1st place thread https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022 ; repo with matrix definitions https://github.com/Fedorov-Artem/kaggle_OTTO/blob/main/README.md ; repo with importances https://github.com/nicolaivicol/otto-recommender ), Yandex Cup 2022 2nd place ( https://github.com/greenwolf-nsk/yandex-cup-2022-recsys ), KDD Cup 2023 10th ( https://arxiv.org/abs/2310.07281 ).
- How: count item pairs that co-occur inside one user's click stream within a time window (5 min ... 24 h), weight each pair by a decay in the time gap and by the position of the later item; build several relation types (any-order co-click, "B anywhere after A", "B immediately after A", "B immediately before A", and here also "candidate exposed in a slate where history item was clicked"); normalise counts by the popularity of the pair members (OTTO repo: `click_to_click_perc_pop`, i.e. count / item popularity, was the top feature). Score a candidate by aggregating over ALL 20 history anchors with position weights 1/(rank+1) (Yandex Cup: +10% recall, +30% precision vs flat), reporting sum / max / mean / count-of-nonzero and the candidate's rank under each matrix.
- Boundary vs CLOSED: closed items were directed last-item transitions and plain co-click ItemKNN; the new parts are the time/position decay, the multi-anchor aggregation with ranks, the popularity (lift/PMI) normalisation, and the exposure->click relation type. Swing-style debiasing (penalise user pairs that share many items; Alibaba https://arxiv.org/abs/2010.05525 ; KDD Cup 2020 Debiasing winner https://github.com/aister2020/KDDCUP_2020_Debiasing_1st_Place ) is the same idea for the popularity-skewed negative problem.
- Transfer / cost: 20-click timestamped histories are exactly the input; ~430k training clicks give a few million pairs, sparse 50k x 50k; minutes, <500 MB.

### F2. Catalogue-level rank-under-retriever features (candidate-generation stage as features)
- Source: Yandex Cup 2022 2nd place (score and rank from every candidate engine as features; https://github.com/greenwolf-nsk/yandex-cup-2022-recsys ), OTTO 6th place "covisitation rank" ( https://www.kaggle.com/c/otto-recommender-system/discussion/384120 ), H&M 2022 1st place "features mostly derived from the retrieval strategies" (secondary: https://speakerdeck.com/kfujikawa/h-and-m-personalized-fashion-recommendation , original https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070 ), KDD Cup 2023 10th (item2vec top-k membership as a feature).
- How: for each retriever already available (ALS, SVD, covisitation, popularity-in-window, user-kNN, the choice model), score the whole 50k catalogue for the user, then record for each slate candidate: its rank among 50k, its score percentile, the gap to the user's top-1 score, and the number of retrievers whose top-K contains it. Ranks act as a per-user "contextual normalisation" (the FeatureSalad RecSys 2024 idea, https://github.com/recsyspolimi/recsys-challenge-2024-ekstrabladet ).
- Boundary vs CLOSED: slate-relative features are closed; these are catalogue-relative, which is different information (how exceptional the candidate is for this user across all items, not across the 10 shown).
- Transfer / cost: 53k x 50k dense dot products per factor model, chunked numpy, a few minutes each; sparse for covisitation.

### F3. User-user / session-kNN neighbourhood scores (V-SKNN, STAN, VSTAN)
- Source: Ludewig & Jannach et al. "Empirical analysis of session-based recommendation algorithms" ( https://arxiv.org/abs/1910.12781 ): tuned kNN beats most neural session models; STAN (Garg et al., SIGIR 2019, https://dl.acm.org/doi/10.1145/3331184.3331322 ) adds three decays; KDD Cup 2020 Debiasing winner uses User-CF alongside Item-CF.
- How: represent each user by the set of their clicks (optionally the 20-click window at event time) with exponential decay on item recency; find the K (50-200) most similar users by cosine/Jaccard, weight neighbours by the recency of their activity; score candidate = sum over neighbours of similarity x indicator(neighbour clicked candidate) x position decay of that click in the neighbour's history. Emit score, rank, number of neighbours that clicked it, and the same computed on neighbours' exposures (how often similar users were shown it).
- Boundary vs CLOSED: ItemKNN is item-item; this is user-user; EASE and ALS are global factorisations, not local neighbourhoods.
- Transfer / cost: user x item sparse matrix with ~430k non-zeros; neighbour search by chunked sparse products; 5-20 min, <1 GB.

### F4. Full-sequence SASRec-style retriever trained with (logQ-corrected) sampled softmax over the catalogue
- Source: OTTO 1st place: the NN session-encoder recall scores were the single largest ablation gain (~+0.005 on the metric; secondary summary above); KDD Cup 2023 NVIDIA winners used transformers plus covisitation ( https://openreview.net/pdf?id=J3wj55kK5t , listing at https://kddcup23.github.io ); Klenitskiy & Vasilev 2023 ( https://arxiv.org/abs/2309.07602 ) show SASRec with full/large-sample cross-entropy beats BERT4Rec; gSASRec ( https://arxiv.org/abs/2308.07192 ) shows BCE with one negative is badly over-confident, many negatives per positive are needed; logQ correction for in-batch negatives (Yi et al. 2019, https://research.google/pubs/pub48840/ ; revisited https://arxiv.org/abs/2507.09331 ); RecSys 2025 4th place uses next-item sampled softmax with 8192 in-batch + 8192 uniform negatives ( https://arxiv.org/abs/2508.06970 ).
- How: build each user's full chronological click sequence (all training clicks, not just the 20-click window), train a 1-2 layer causal transformer (dim 48-64, seq len 50) to predict the next click with sampled softmax (256-1024 negatives), negatives drawn from the EXPOSURE distribution (what the simulator shows) with and without logQ correction; use as features the candidate's logit, its rank among 50k, and the cosine between candidate and sequence embedding at the event timestamp.
- Boundary vs CLOSED: closed items were residual slate-level DIN/GRU/attention models and a 10-candidate softmax choice model; this is a stand-alone catalogue-level next-click retriever whose outputs feed CatBoost. Note the 50k-way objective matches what a recommender simulator typically does when it picks what to show.
- Transfer / cost: 3.2M embedding params; ~430k training positions; PyTorch CPU 30-90 min for 10-20 epochs, <2 GB.

### F5. Explicit model of the logging recommender (exposure propensity) and "surprise" residuals
- Source: Schnabel et al. 2016 "Recommendations as Treatments" (propensity = P(item exposed | user), https://arxiv.org/abs/1602.05352 ); Luo et al. SIGIR 2024 "Logging-Policy-aware Propensity" learns a map from user-item features to logging-policy scores ( https://www.cse.lehigh.edu/~brian/pubs/2024/SIGIR/ ); Oosterhuis & de Rijke policy-aware estimators for top-k ( https://arxiv.org/abs/2005.09035 ); impression-aware survey ( https://arxiv.org/abs/2308.07857 ) lists 10+ works using impression counts/positions as features.
- How: train a cheap classifier (logistic or small GBDT) on "was this item in the slate" with the 10 exposed items as positives and sampled catalogue items as negatives, using popularity, history-similarity, recency, and history-length features; then feed (a) the exposure logit, (b) exposure logit minus popularity-only logit ("why was this unpopular item shown?" = the recommender's personalised score), (c) the candidate's exposure rank in the user's catalogue, and (d) interactions of these with the click-model score. Also use the fitted propensity to draw hard negatives for F4.
- Boundary vs CLOSED: exposure-aware ALS used exposures as confidence weights inside a factor model; this is a discriminative policy model whose outputs are features and a sampler.
- Transfer / cost: 6.9M exposure rows + equal sampled negatives, LightGBM/CatBoost minutes; <2 GB.

### F6. Item lifecycle / trend / timeliness features, including same-day transductive exposure counts
- Source: RecSys Challenge 2024 (EB-NeRD news): organisers' paper ( https://arxiv.org/abs/2409.20483 ) reports that removing future-information popularity features dropped the winner (team :D) from 88.64 to 76.99 AUC and other top teams by 2-7 AUC points; 3rd place (Tom3TK) built on "article timeliness"; OTTO repos use item counts per day / same day / week; Yandex Cup 2022 computes popularity only from recent sessions.
- How: for each candidate at event time t compute exposures, clicks and smoothed CTR in trailing windows (1 h, 6 h, 24 h, 3 d), trend ratios (clicks last 24 h / previous 24 h), item age (time since first exposure), time since last click anywhere, share of lifetime clicks so far, and the item's exposure share on the SAME calendar day including later events (available on the test day because all test slates are known). Normalise same-day counts by that day's event count because the test day holds only first events.
- Transfer / cost: a simulator that re-ranks by popularity makes trend and same-day exposure share highly informative; groupbys over 6.9M rows, minutes.

### F7. Graph embeddings and multi-hop covisitation (ProNE / node2vec / LightGCN; 2-hop scores)
- Source: H&M 1st place used ProNE graph-embedding similarity as its main candidate source (secondary, links in F2); OTTO 2nd place used ProNE session-item similarity; KDD Cup 2020 Debiasing winner used multi-hop i2i graph modelling; OTTO 1st used "beam-search-like repeated lookups" of covisitation; LightGCN ( https://arxiv.org/abs/2002.02126 ), node2vec ( https://arxiv.org/abs/1607.00653 ), ProNE ( https://www.ijcai.org/proceedings/2019/594 ).
- How: build the bipartite user-item click graph (and/or the item-item covisitation graph from F1), compute spectral embeddings (ProNE: sparse SVD then spectral propagation) or random-walk skip-gram embeddings; features = cosine of candidate with each history item (max/mean/last-3), cosine with the mean user vector, and a 2-hop covisitation score (candidate's covisitation with the items that covisit with the history).
- Transfer / cost: multi-hop smooths sparsity of 20-click histories; ProNE on 103k nodes in minutes; node2vec via gensim 10-30 min; <2 GB.

### F8. Sequence-window item2vec embeddings (short vs long context windows)
- Source: OTTO repos (two word2vec models, short window for clicks, long window for purchases; `w2v_5_max` among top gain features, https://github.com/Fedorov-Artem/kaggle_OTTO/blob/main/README.md ); KDD Cup 2023 10th place item2vec per locale; Dressipi 2022 Layer6 co-occurrence features ( https://www.cs.toronto.edu/~mvolkovs/Recsys_2022.pdf ).
- How: train skip-gram on per-user click sequences with window 2-3 (sequential) and window 20 (bag), optionally on exposure sequences too; features = similarity of candidate to last-1/last-3/last-5/all-20 clicks (max, mean, min) per model.
- Transfer / cost: corpus is small (~430k tokens), so use many epochs and dim 32-64; gensim minutes. Lower ceiling than F1/F7 but nearly free.

### F9. User-cluster conditional popularity and smoothed target encodings (co-clustering)
- Source: OTTO repo `src_pop_cl50` (sessions clustered by KMeans on embedding; cluster popularity as candidate source and feature, https://github.com/nicolaivicol/otto-recommender ); H&M 2nd place popularity grouped by demographic buckets (secondary); NVIDIA RecSys 2020/2021 winners: k-fold smoothed target encoding of feature combinations ( https://medium.com/rapids-ai/winning-solution-of-recsys2020-challenge-gpu-accelerated-feature-engineering-and-training-for-cd67c5a87b1f ).
- How: KMeans (50-300 clusters) on ALS/SVD user vectors or on F4 sequence embeddings; per (cluster, item) compute exposure count, click count, Bayesian-smoothed CTR with out-of-fold encoding; also encode (item x history-length bucket) and (item x hour-of-day) CTRs since the evaluation weights by history length.
- Transfer / cost: seconds for clustering, minutes for encodings.

### F10. Long-term vs short-term interest split (multi-window similarity aggregates)
- Source: RecSys 2024 2nd place (BlackPearl) "Hierarchical User Interest Modeling" separating stable and fast-changing interests (organisers' paper above); OTTO features split by last-5 vs last-20 and 3-hour cut-offs; Yandex Cup 2022 statistics over last 10 vs last 50 actions.
- How: compute every history-similarity feature (ALS, SVD, F1, F7, F8) separately over last-1, last-3, last-5 clicks, the clicks within the last 1 h / 24 h, and all 20; add differences and ratios (short minus long) and the position-decayed version.
- Transfer / cost: trivial once the base similarities exist; some overlap with existing "recency fractions", so expect small but cheap gains.

### F11. Impression-discounting functions over user x item exposure history (adjacent to closed, listed for completeness)
- Source: Lee et al. KDD 2014 "Modeling impression discounting in large-scale recommender systems" ( https://researchr.org/publication/LeeLTS14 ): discount = f(days since last impression, impression count, last position) with linear/inverse/exponential/quadratic forms; Ma et al. 2016 fatigue features in LambdaMART (via survey above).
- How: for each candidate, days since last shown to this user and count of prior shows, combined as exp(-a*days)*count etc.
- Boundary: raw "ignored presentation" counts are CLOSED; only the decayed functional forms and their interaction with global item trend (F6) are new. Lowest priority.

Not found: no competition write-up documents "sequence of slates / dwell patterns" beyond timing features (closed), nor explicit co-clustering of items by exposure pattern beyond F9; no public description of this competition's simulator was found.

## Part 2 - Ranked top-5 by expected value for this task

1. F6 Item lifecycle / trend + same-day exposure share: in the only comparable impression-based AUC challenge (RecSys 2024) popularity-trend features were worth 2-12 AUC points and the test-day slates are fully observable here; cheapest to build.
2. F4 Full-sequence SASRec retriever with exposure-sampled, logQ-corrected softmax: the largest single ablation gain reported by the OTTO winner and the KDD Cup 2023 winner's backbone, and it is structurally different from the closed residual NNs and the 10-way choice model.
3. F1 Decayed, multi-anchor, popularity-normalised covisitation: consistently the top-gain feature family across OTTO/Yandex/KDD'23 repos, and the lift/exposure->click normalisation directly addresses popularity-skewed negatives.
4. F5 Logging-policy propensity model and surprise residuals: the negatives are chosen by a recommender, so a model of "why was it shown" isolates the recommender's own relevance estimate; cheap and also yields hard negatives for F4.
5. F3 User-user session-kNN: a strong, local, non-factorised signal that no closed item covers, repeatedly competitive with neural models in the Jannach benchmarks, and 10-20 minutes to compute.

Runners-up: F2 rank-under-retriever (very cheap, add alongside F3), F7 graph/multi-hop embeddings.
