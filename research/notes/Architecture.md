# Model Architecture

The `MatchupModel` is a Siamese neural network with two major sections:
1.  **Shared Deck Encoder (Deep Sets)**: Encodes an unordered set of 8 cards into a fixed-dimensional deck representation.
2.  **Symmetric Matchup Head (Bradley-Terry)**: Estimates win probabilities based on deck representations, guaranteeing symmetry and anti-symmetry.

## 1. Shared Deck Encoder
To guarantee that the card ordering in a deck does not change the deck embedding, we implement a Deep Sets representation layer:
$$v_{\text{deck}} = \frac{1}{8} \sum_{i=1}^{8} \text{MLP}(E(c_i))$$
where:
*   $E(c_i)$ is a 16-D trainable embedding vector for card ID $c_i$.
*   $\text{MLP}$ is a shared projection layer (Linear $\rightarrow$ LayerNorm $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear) projecting card representations to $D_{\text{proj}} = 16$.

## 2. Bradley-Terry Interaction Head
The intrinsic advantage score $\theta(A, B)$ is parameterized as a bilinear form:
$$\theta(A, B) = v_A^T W v_B$$
To satisfy anti-symmetry ($\theta(A, B) = -\theta(B, A)$), the matrix $W$ is constrained to be skew-symmetric by design:
$$W = M - M^T$$
where $M$ is a square matrix of size $D_{\text{proj}} \times D_{\text{proj}}$.

## 3. Skill Confounder Module
Player skill differences are modeled using an odd function $f(\Delta T)$ of the relative trophies $\Delta T$:
$$f(\Delta T) = \frac{\text{MLP}(\Delta T) - \text{MLP}(-\Delta T)}{2}$$
Because $f(-\Delta T) = -f(\Delta T)$, the skill bias behaves as an odd function. At inference, we set $\Delta T = 0$, ensuring $f(0) = 0$ exactly, removing the confounder bias and isolating the deck matchup score.
