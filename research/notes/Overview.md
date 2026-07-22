# Project Overview

The objective of this project is to build an intrinsic matchup prediction model for Clash Royale. In competitive evaluation, understanding card interaction advantages is difficult because player skill differences confound the outcome. 

This project solves this by using a Siamese neural network to un-confound player skill and isolate pure card interaction advantage.

## Problem Scope
Given two decks of 8 cards each and a proxy of player skill difference (relative trophy differences):
*   We model relative trophies as a confounder bias during training.
*   At inference, we remove the skill bias by setting trophy difference to zero, isolating the intrinsic card advantage score.
*   The model guarantees that symmetric swaps satisfy mathematical constraints:
    *   $P(A, A) = 50.0\%$ exactly.
    *   $P(A, B) + P(B, A) = 1.0$ exactly.
