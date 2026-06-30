# 🔴🟡 Connect 4 — Human vs AI (Minimax / Alpha-Beta)

A graphical **Connect 4** game built with `pygame`, where a human player competes against an AI opponent powered by the **Minimax algorithm** (with optional **Alpha-Beta pruning**).

---

## 📌 Overview

This project implements the classic Connect 4 game with a twist on win detection and AI decision-making:

- The **AI** searches future moves using Minimax / Alpha-Beta pruning up to a configurable depth
- The board doesn't end the moment a 4-in-a-row appears — play **continues until the board is completely full**
- The final winner is decided by **counting all distinct connect-four groups** on the board using a BFS-based connected-components search, not just "first to connect 4"

---

## 🎮 How It Works

### Game Flow
1. Player drops a piece by clicking a column
2. AI responds by running Minimax (or Alpha-Beta) search and dropping its own piece
3. This continues until the **board is full**
4. The winner is the player with **more total connect-four groups** anywhere on the board (`strict_count_fours`)

### AI Decision Making
- **Minimax** explores the game tree to a fixed depth, maximizing the AI's score and minimizing the player's
- **Alpha-Beta Pruning** (optional) skips branches that can't affect the final decision, making the search faster at the same depth
- Each AI move prints to the console:
  - The full search tree (`print_minimax_tree`)
  - Total **nodes expanded**
  - **Time taken** for the search

### Heuristic Evaluation
When the search depth limit is reached without a guaranteed win/loss, the board is scored using `score_position`:
- Rewards center-column control
- Scores sliding 4-cell windows (horizontal, vertical, both diagonals)
- Heavily rewards 3-in-a-row with an open spot, and penalizes the opponent having the same

---

## ⚙️ Configuration

At the top of the file:

```python
USE_ALPHA_BETA = False   # True → Alpha-Beta pruning, False → plain Minimax
DEPTH = 3                # Search depth (higher = smarter but slower AI)
```

---

## 🛠️ Tech Stack

`Python` · `pygame` · `NumPy`

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install pygame numpy
   ```
2. Run the game:
   ```bash
   python Connect4.py
   ```
3. Click a column to drop your piece (red). The AI (yellow) responds automatically.
4. Keep playing — the game only ends when the **board is completely full**, then the winner is announced based on total connect-four groups.

---

## 🧠 Key Concepts Demonstrated

- **Adversarial search**: Minimax with maximizing/minimizing players
- **Alpha-Beta pruning**: search space reduction without affecting the optimal result
- **Heuristic design** for non-terminal board states
- **Graph traversal (BFS)** for detecting connected groups of pieces, used to determine the final winner under the "fill the whole board" rule
- **Real-time game state visualization** with `pygame`, including animated piece drops

---

## 📋 Notes

- Console output during AI turns is useful for debugging/understanding AI decisions — check the terminal while playing
- Increasing `DEPTH` significantly increases computation time, especially with plain Minimax (no pruning)
- Switching `USE_ALPHA_BETA = True` gives identical move quality to Minimax but explores fewer nodes
