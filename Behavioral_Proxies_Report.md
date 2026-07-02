# Behavioral Proxy Construction Report

The **TraitAlign** framework relies on engineered behavioral proxies—metrics derived directly from a user's interaction history—to act as alignment targets for the causal intervention objectives. 

Below is a detailed breakdown of how these proxies are constructed for each of the three datasets evaluated in the codebase.

---

## 1. MovieLens 1M Dataset
The proxies for the standard MovieLens 1M dataset are pre-computed in the `kaggle_master_prep.py` script. The construction focuses heavily on metadata (genres, release years, and popularity).

For every user, the script groups their historical interactions and calculates the following metrics:
*(Note: MiniLM embeddings are **not** used to calculate these MovieLens 1M behavioral proxies. The script does use MiniLM later to encode user occupations as metadata, but the proxies themselves are built purely from interaction logic).*
- **Nostalgia (`raw_nostalgia`)**: The mean release year of all movies interacted with by the user. Lower values indicate a preference for older classic films, while higher values indicate a preference for modern films.
- **Mainstream Preference (`raw_mainstream`)**: The mean global popularity (total interaction count across all users) of the movies the user watched.
- **Diversity (`raw_diversity`)**: The Shannon entropy of the user's genre distribution. High entropy indicates the user watches a wide, evenly distributed variety of genres.
- **Adrenaline Preference (`adrenaline_pref`)**: The fraction of the user's total interactions that fall into high-arousal genres: `{'Action', 'Thriller', 'Horror', 'Sci-Fi'}`.
- **Average Rating (`raw_rating`)**: The mean rating the user assigns to movies.

*Normalization*: All metrics (except the fraction-based `adrenaline_pref`) are min-max scaled between `[0, 1]` across the entire user population.

---

## 2. LastFM 360K Dataset
The LastFM dataset lacks detailed item metadata (like genres or release years) and instead provides streaming play counts. Therefore, the proxies in `kaggle_master_prep.py` are heavily focused on consumption volume and artist discovery.

For every user, the script groups their listening history and calculates:
*(Note: Similar to MovieLens, MiniLM embeddings are **not** used to calculate these behavioral proxies. MiniLM is used to encode LastFM country data as user metadata, but the proxies rely purely on play counts).*
- **Engagement Volume (`raw_engagement`)**: The sum total of all plays across all artists by the user.
- **Discovery Rate (`discovery_rate`)**: The number of unique artists the user listened to, divided by their total plays. A high discovery rate indicates someone who samples many artists rather than looping the same songs.
- **Mainstream Preference (`raw_mainstream`)**: The weighted average of global artist popularity. The global plays of an artist are weighted by the specific user's play count for that artist.
- **Diversity (`raw_diversity`)**: The Shannon entropy of the user's play distribution over artists. High entropy means their listening is spread evenly across many artists, rather than concentrated heavily on just a few.
- **Long-Tail Preference (`long_tail_pref`)**: The fraction of a user's total plays that belong to artists *outside* the top 10% most popular artists globally.

*Normalization*: Similar to MovieLens, these features are aggregated and scaled to prepare them for the alignment head.

---

## 3. Personality 2018 Dataset
Unlike the other two datasets which are pre-processed by the prep script, the behavioral proxies for Personality 2018 are **dynamically constructed at runtime** inside the `traitalign_kaggle_trainer.ipynb` notebook (because it aligns the original 5-dimensional Big Five OCEAN traits directly). 

The notebook first dynamically merges the `movies.csv` from the MovieLens 25M dataset to attach genre labels to the Personality 2018 ratings. Then, it calculates exactly 5 behavioral proxies intended to loosely map to the 5 psychological dimensions:

1. **Diversity Index**: Using the 384-dimensional MiniLM embeddings (`P_i.npy`) for the movies the user interacted with, the script calculates the mean pairwise cosine distance between all pairs of movies. High distance indicates high structural diversity.
2. **Novelty Preference**: Calculated as the mean of `-log(P(item))`, where `P(item)` is the normalized popularity probability of the item. This measures how often the user consumes rare/obscure items.
3. **Temporal Stability**: The user's interaction history is chopped into chronological chunks of 50 interactions. The script computes the Spearman rank correlation between the genre distributions of adjacent chunks, measuring how stable the user's preferences remain over time.
4. **Exploration Rate**: The script identifies the user's personal Top-3 most frequently watched genres. The exploration rate is the fraction of their interactions that occur *outside* of these dominant Top-3 categories.
5. **Cross-Category Reach**: The Shannon entropy of the user's genre distribution over all their historical interactions.

*Causal Intervention*: After these 5 proxies are calculated and min-max scaled, the notebook performs **Stratified Matching** directly on them. It bins users into deciles by graph degree (interaction count), and calculates the difference between a user's raw proxy and the mean proxy of their matched degree bin. This de-confounded residual becomes the final target for the model.
