# Map Search Ranking with Learning to Rank algorithm

Trained a **LambdaMART** learning-to-rank model over
a **real restaurant dataset**, then integrated it with browser UI that runs the actual trained model
client-side.

```
A-Map-Search-Ranking-System/
├── data/
│   ├── real_geoplaces2.csv        raw UCI dataset (places, real GPS coords)
│   ├── real_chefmozcuisine.csv    raw UCI dataset (real cuisine tags)
│   ├── real_chefmozhours4.csv     raw UCI dataset (real weekly opening hours)
│   ├── real_rating_final.csv      raw UCI dataset (real consumer ratings)
│   ├── prepare_real_data.py       cleans/merges the above -> restaurants_real.json
│   ├── generate_sessions.py       simulates search sessions over the real catalog
│   └── restaurants_real.json      70 real restaurants, cleaned
├── src/
│   ├── train.py                    trains the LambdaMART ranker (LightGBM)
│   └── export_for_ui.py            exports trained trees + real catalog for the UI
├── models/                         trained model (native + JSON + minimal JSON)
├── static/restaurants.json         real restaurant catalog, UI-ready
└── index.html                       the finished, self-contained demo - open this
```

## What's real vs. simulated

| Piece | Source |
|---|---|
| Restaurant names, GPS coordinates, cuisines, prices, weekly hours | **Real.** UCI "Restaurant & Consumer Data" (Vargas-Govea et al., 2011), San Luis Potosí, Mexico. |
| Star ratings, review counts | **Real.** Aggregated from real diners' real ratings in the same dataset. |
| Search queries and relevance labels used to *train* the ranker | **Simulated.** No public query-log dataset exists for this restaurant catalog, so training sessions (user location, query, relevance judgment) are generated the same way a company would bootstrap a ranker before it has click logs - see `generate_sessions.py`. |
| The query *you* type into the demo | **Real, yours.** Matched live against the real names/cuisines. |

## Run it yourself

```bash
pip install lightgbm scikit-learn pandas numpy
python3 data/prepare_real_data.py   # clean/merge real UCI CSVs -> restaurants_real.json
python3 data/generate_sessions.py   # simulate training sessions over the real catalog
python3 src/train.py                # train models/lambdamart.{txt,json}
python3 src/export_for_ui.py        # export models/model_min.json + static/restaurants.json
```

Then just open the pre-built `index.html` directly - no server needed.

## The algorithm: LambdaMART

`train.py` trains **LambdaMART** - gradient-boosted trees (LightGBM's
`LGBMRanker`, `objective="lambdarank"`) with a pairwise/listwise ranking loss:
it learns from pairs of candidates *within the same query* ("A should outrank
B") with gradients weighted by how much swapping them would change NDCG.
That directly optimizes ranking order, unlike pointwise regression on
relevance. Rows are grouped by `query_id`, split by query (not row) to avoid
leakage, and evaluated with NDCG@3/5/10 on held-out queries.

Result: **NDCG@5 ≈ 0.955**. `text_match` and `rating` dominate feature
importance, followed by `is_open` and `distance_km` - sensible for a
restaurant search ranker.

search box and a time-of-day picker.

