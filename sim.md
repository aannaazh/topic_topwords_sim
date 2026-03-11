⏺ 📚 Analogy: A Library Cataloging System

  Imagine you're organizing books in a massive library where:
  - Documents are like bookshelves
  - Sentences are individual books
  - Topics are genres (fiction, science, history...)
  - Words are the pages within books

  But there's a twist: the pages got stuck together, so you need to figure out where one word ends and another begins (word segmentation), while also determining which genre each book belongs to (topic modeling).

  This code simulates that process in a Chinese text setting!

  ---
  📊 System Architecture

  ┌─────────────────────────────────────────────────────────────────┐
  │                     TopWordsTopicSimulator                       │
  │                  (The "Book Factory" - generates data)          │
  ├─────────────────────────────────────────────────────────────────┤
  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
  │  │ Generate│    │  Words  │    │  Mix    │                     │
  │  │Vocabulary│   │from Topics│   │background│                     │
  │  └────┬────┘    └────┬────┘    └────┬────┘                     │
  │       │              │              │                            │
  │       └──────────────┴──────────────┘                            │
  │                      │                                           │
  │                      ▼                                           │
  │               ┌────────────┐                                     │
  │               │   Corpus   │  (Documents → Sentences → Words)  │
  │               └─────┬──────┘                                     │
  └────────────────────│────────────────────────────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                      TopWordsTopicModel                           │
  │                (The "Cataloger" - learns from data)              │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                   │
  │   ┌─────────────────┐              ┌─────────────────┐         │
  │   │  E-Step         │              │   M-Step        │         │
  │   │  (Inference)    │ ─────────▶   │  (Update)       │         │
  │   └─────────────────┘              └─────────────────┘         │
  │          │                               │                       │
  │          ▼                               ▼                       │
  │   ┌─────────────────┐              ┌─────────────────┐         │
  │   │ Inside-Outside  │              │   Update φ      │         │
  │   │   Algorithm     │              │   Update θ      │         │
  │   │                 │              │   Update π      │         │
  │   └─────────────────┘              └─────────────────┘         │
  │                                                                   │
  │   Parameters:                                                     │
  │   φ (phi)   ──→  Topic-to-Word distribution                       │
  │   θ (theta)  ──→  Document-to-Topic distribution                  │
  │   π (pi)     ──→  Mix: topic-words vs background-words          │
  │                                                                   │
  └─────────────────────────────────────────────────────────────────┘

  ---
  🚶 Step-by-Step Walkthrough

  Part 1: The Simulator (lines 10-82)

  The simulator creates synthetic Chinese text data:

  1. Creates a vocabulary (_generate_vocabulary, lines 34-58):
    - Generates single-character words (30% of words by default)
    - Generates multi-character words using a Zipf-like distribution (shorter words more common)
    - Uses Chinese characters from Unicode range 0x4e00 onwards
  2. Generates a corpus (generate_corpus, lines 60-82):
  For each document:
      Sample θ ~ Dirichlet(α)  ← How much of each topic in this doc
      For each sentence:
          Sample topic z ~ θ      ← Which topic generates this sentence
          Sample π ~ Beta(γ)       ← How "topic-focused" this sentence is
          For each word in sentence:
              Pick source: topic z with probability π, else background
              Sample word from chosen source's distribution

  Part 2: The Model (lines 88-299)

  The model learns to recover the hidden topics and segment words:

  Inside-Outside Algorithm (_inside_outside, lines 100-138)

  This is the heart of the word segmentation. Given a continuous string like "我喜欢中文学习", it computes:
  - f[i] (forward/inside): log-probability of segmenting prefix [0:i]
  - b[i] (backward/outside): log-probability of segmenting suffix [i:L]
  - Expected count of each possible word at each position

  Example: "机器学习" (jī qì xué xí)
  Forward pass finds all possible segmentations:
    - [机器][学习] ✓
    - [机][器][学习]
    - [机器][学][习]
    - ...and computes probabilities for each

  Backward pass computes outside probabilities

  Result: "机器" has 0.8 probability, "学习" has 0.85 probability

  EM Training Loop (train, lines 162-238)

  For each iteration:

      E-Step (Compute expectations):
          For each sentence and each possible topic:
              Run Inside-Outside → get word counts
              Compute posterior P(topic | sentence)
              Compute source posteriors P(topic-source | word)

      M-Step (Update parameters):
          φ[k,w] = (count(w from topic k) + β) / (total + V·β)
          θ[d,k] = (topic k count in doc d + α) / (total + K·α)
          π = (special count + γ₁) / (total + γ₁ + γ₂)

      Evaluate and track metrics

  Evaluation (evaluate, lines 240-273)

  Uses the Hungarian algorithm to match predicted topics to ground truth topics (since topic labels are arbitrary).

  ---
  ⚠️ Gotcha: The "Label Switching" Problem

  A subtle but critical issue in topic modeling: the model learns topic K, but it might correspond to ground truth topic 2, 1, or any other. There's no inherent order!

  # This is why we need the Hungarian algorithm (line 252):
  row_ind, col_ind = linear_sum_assignment(-confusion)

  # confusion might look like:
  #         GT:0  GT:1  GT:2
  # Pred:0   5     20    3    ← "Topic 0" is actually GT topic 1!
  # Pred:1  30     2     1
  # Pred:2   1     4    25

  Without this mapping step, your accuracy would look terrible even if the model learned perfect topics!

  ---
  🔑 Key Parameters

  | Parameter | Meaning                            | Typical Range |
  |-----------|------------------------------------|---------------|
  | α (alpha) | Topic sparsity per doc             | 0.1-1.0       |
  | β (beta)  | Word sparsity per topic            | 0.01-0.1      |
  | γ (gamma) | Background vs topic word mix       | (2, 2)        |
  | K         | Number of topics                   | 2-20          |
  | π         | P(word from topic, not background) | 0-1           |

  ---
  📈 What the Code Actually Does

  When you run this (line 318 or 326-327), it:
  1. Generates 1000 synthetic documents with 10 sentences each
  2. Trains the model for 10 iterations using EM
  3. Tracks: log-likelihood (should increase), segmentation F1, topic accuracy
  4. Saves a plot of training metrics to training_metrics.png

  The plot helps you see if the model is converging and learning properly!