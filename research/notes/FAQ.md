# Frequently Asked Questions (FAQ)

### Q1: Why are card level structures ignored? (e.g. King Tower levels)
King tower levels are omitted from parsing (stored as NULL in SQLite) to reduce confounding variance, since player level is strongly correlated with trophies.

### Q2: How is the Elixir cost of the Mirror card handled?
The Mirror card replicates the previously played card for +1 Elixir. During dataset preparation, its Elixir cost defaults to the deck's average Elixir cost to avoid dynamic calculation issues.

### Q3: Why is skill difference modeled as an odd function?
To satisfy mathematical anti-symmetry under trophy adjustments:
$$P(A, B \mid \Delta T) + P(B, A \mid -\Delta T) = 1.0$$
If skill difference bias is an odd function ($f(-\Delta T) = -f(\Delta T)$), this holds exactly. Setting $\Delta T = 0$ removes skill corrections completely.

### Q4: Why is the Jaccard similarity index of embeddings low?
Average card pooling introduces rotational invariance. The model preserves relative distances (topology) to predict matchups, but absolute weights undergo coordinate rotations between random seeds.
