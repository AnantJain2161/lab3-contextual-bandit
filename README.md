# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Course:** Reinforcement Learning Fundamentals  
**Student Name:** Anant Jain 
**Roll Number:** U20230155  


---

## Executive Summary

This project implements a comprehensive **Contextual Multi-Armed Bandit (CMAB)** system for personalized news article recommendations. The system combines user classification with three distinct bandit algorithms (Epsilon-Greedy, UCB, and SoftMax) to maximize user engagement through intelligent exploration-exploitation strategies.

### 🎯 Key Results

- **Best Algorithm:** Upper Confidence Bound (UCB) with C=2.0
- **Best Average Reward:** 8.4585
- **User Classification Accuracy:** 90.0%
- **Time Horizon:** 10,000 steps
- **Total Arms:** 12 (3 user contexts × 4 news categories)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Analysis and Insights](#analysis-and-insights)
6. [Implementation Details](#implementation-details)
7. [Conclusions](#conclusions)
8. [How to Run](#how-to-run)
9. [File Structure](#file-structure)

---

## Introduction

### Contextual Bandit Framework

A **Contextual Bandit** extends the Multi-Armed Bandit problem by incorporating context information:

- **Context (State):** User category (User1, User2, User3)
- **Arms (Actions):** News categories (Entertainment, Education, Tech, Crime)
- **Reward:** Engagement signal from the sampler
- **Goal:** Learn policy π(arm | context) that maximizes expected cumulative reward

### Arm Mapping

The system uses a flattened 12-arm representation:

| Arm Index | User Context | News Category |
|-----------|--------------|---------------|
| 0-3       | User1        | Entertainment, Education, Tech, Crime |
| 4-7       | User2        | Entertainment, Education, Tech, Crime |
| 8-11      | User3        | Entertainment, Education, Tech, Crime |

---

## Problem Statement

The assignment required building a recommendation system that:

1. **Classifies users** into contextual categories (User1, User2, User3)
2. **Maps news articles** to arms (4 categories per user type)
3. **Implements three bandit algorithms** to optimize article recommendations
4. **Maximizes cumulative rewards** over 10,000 time steps

---

## Methodology

### 1. Data Preprocessing

#### Dataset Overview

- **News Articles:** 209,527 articles with 6 features (link, headline, category, description, authors, date)
- **Train Users:** 2,000 users with 33 features and labels (user_1, user_2, user_3)
- **Test Users:** 2,000 users with 32 features (no labels)

#### Preprocessing Steps

1. Handled missing values using **median imputation** for numeric columns
2. Identified categorical features: `region_code`, `browser_version`
3. Applied **Label Encoding** to categorical features with unknown value handling
4. Created user categories using **K-Means clustering** (3 clusters)
5. Normalized feature distributions across train and test sets

#### Data Splits

- **Training:** 1,600 users (80%)
- **Validation:** 400 users (20%)

### 2. User Classification Model

**Classifier:** Random Forest Classifier

- Number of estimators: 100
- Max depth: 10
- Training samples: 1,600
- Validation samples: 400

#### Performance Metrics

```
                precision    recall  f1-score   support
        user_1       0.89      0.86      0.87       142
        user_2       0.97      0.89      0.93       142
        user_3       0.84      0.97      0.90       116
    ──────────────────────────────────────────────────
    accuracy                           0.90       400
   macro avg       0.90      0.90      0.90       400
weighted avg       0.90      0.90      0.90       400
```

**Key Insight:** The classifier achieves **90% accuracy**, ensuring reliable context detection for the recommendation engine.

### 3. Contextual Bandit Algorithms

#### 3.1 Epsilon-Greedy Strategy

**Algorithm:**
- With probability ε: select a random arm (exploration)
- With probability (1-ε): select the arm with highest Q-value (exploitation)
- Q-values updated using incremental averaging

**Hyperparameter Tuning:** Tested ε ∈ {0.01, 0.1, 0.3}

#### 3.2 Upper Confidence Bound (UCB)

**Algorithm:**
- Selects arm that maximizes: Q(a) + C√(ln(t)/N(a))
- First term: estimated reward
- Second term: optimistic bonus for exploration
- Balances exploration automatically based on visit counts

**Hyperparameter Tuning:** Tested C ∈ {0.5, 1.0, 2.0}

#### 3.3 SoftMax Strategy

**Algorithm:**
- Selects arm based on softmax probability distribution
- Probability of arm a: P(a) = exp(Q(a)/τ) / Σ exp(Q(i)/τ)
- Temperature τ controls exploration smoothness

**Configuration:** Temperature τ = 1.0

---

## Results

### Performance Comparison

| Algorithm       | Hyperparameter | Avg Reward | Std Dev | Final Reward |
|----------------|----------------|------------|---------|--------------|
| Epsilon-Greedy | ε=0.01         | **8.0218** | 1.9591  | 9.9762       |
| Epsilon-Greedy | ε=0.1          | 7.6760     | 3.1889  | 9.5790       |
| Epsilon-Greedy | ε=0.3          | 6.1314     | 4.7849  | 10.2141      |
| UCB            | C=0.5          | 4.1748     | 0.8231  | 3.8203       |
| UCB            | C=1.0          | 4.1862     | 0.8279  | 4.3460       |
| **UCB**        | **C=2.0**      | **8.4585** | 1.3261  | 8.5287       |
| SoftMax        | τ=1.0          | 7.9382     | 1.7950  | 7.6242       |

### 🏆 Winner: UCB with C=2.0

Achieves the highest average reward of **8.4585** across 10,000 time steps.

### Algorithm-Specific Results

#### Epsilon-Greedy Results

| ε    | Avg Reward | Std Dev | Interpretation |
|------|------------|---------|----------------|
| 0.01 | 8.0218     | 1.9591  | High exploitation, good convergence |
| 0.1  | 7.6760     | 3.1889  | Balanced exploration-exploitation |
| 0.3  | 6.1314     | 4.7849  | High exploration, high variance |

**Key Finding:** Lower ε values provide better average reward by focusing on exploitation once good arms are identified.

#### UCB Results

| C   | Avg Reward | Std Dev | Interpretation |
|-----|------------|---------|----------------|
| 0.5 | 4.1748     | 0.8231  | Too conservative, limited exploration |
| 1.0 | 4.1862     | 0.8279  | Insufficient exploration bonus |
| 2.0 | **8.4585** | 1.3261  | **Optimal balance, best overall** |

**Key Finding:** UCB with C=2.0 provides the highest average reward. The aggressive exploration bonus enables faster convergence to optimal arms.

#### SoftMax Results

| τ   | Avg Reward | Std Dev |
|-----|------------|---------|
| 1.0 | 7.9382     | 1.7950  |

**Key Finding:** SoftMax provides competitive performance with smooth probability-based exploration.

### Recommendation Engine Examples

**Pipeline Implementation:**
1. Classify User → Random Forest classifier → User context
2. Select Category → Apply UCB (C=2.0) → Extract Q-values → Map to news category
3. Recommend Article → Query news database → Return article headline

**Sample Recommendations:**

- **User 1:** Context=user_2, Category=Tech  
  → *"Watch The Top 9 YouTube Videos Of The Week..."*

- **User 2:** Context=user_1, Category=Tech  
  → *"This Is The Robot Dallas Police Used To Kill Shooting Suspect..."*

- **User 3:** Context=user_1, Category=Tech  
  → *"Mailbox App Gets 800,000-Person-Long Waiting List..."*

---

## Analysis and Insights

### 1. Algorithm Comparison

| Algorithm       | Pros | Cons | Best Use Case |
|----------------|------|------|---------------|
| Epsilon-Greedy | Simple, interpretable, consistent | Fixed exploration rate, inefficient early on | Baseline comparison, simplicity required |
| UCB            | Adaptive exploration, best performance | Requires tuning of C parameter | Optimizing reward is critical |
| SoftMax        | Smooth exploration, probabilistic | Moderate performance | Smooth probability distribution preferred |

### 2. Hyperparameter Sensitivity

#### Epsilon-Greedy (ε)
- **ε=0.01:** Best (8.02), high exploitation
- **ε=0.1:** Medium (7.68), balanced
- **ε=0.3:** Worst (6.13), too much exploration
- **Recommendation:** Lower values preferred

#### UCB (C)
- **C=0.5:** Poor (4.17), insufficient exploration
- **C=1.0:** Poor (4.18), insufficient exploration
- **C=2.0:** Excellent (8.46), optimal balance
- **Recommendation:** Higher C better for 12-arm problem; exploration bonus critical

### 3. Convergence Behavior

**Epsilon-Greedy:**
- Convergence period: first 2,000 steps
- Settles to average by step 5,000
- Stable but with exploration noise

**UCB:**
- Very rapid convergence (especially C=2.0)
- C=2.0 converges within first 1,000 steps
- Higher exploration bonus enables faster learning

**SoftMax:**
- Smooth convergence without sharp changes
- Takes ~3,000 steps to stabilize
- Gradual improvement benefits from soft probabilities

### 4. Context-Specific Performance

The Q-values learned by each algorithm vary by user context:

- **User1:** Generally lower rewards across categories
- **User2:** Strong preference for specific categories
- **User3:** Mixed performance, highest variance

This validates the **contextual bandit approach** as different user contexts genuinely have different optimal strategies.

---

## Implementation Details

### Technical Stack

**Language:** Python 3.12

**Libraries:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning models
- `matplotlib` & `seaborn` - Visualization
- `rlcmab_sampler` - Reward sampling

### Code Structure

**Modules Implemented:**

1. `EpsilonGreedyContextual` - Epsilon-greedy strategy
2. `UCBContextual` - Upper confidence bound strategy
3. `SoftMaxContextual` - SoftMax (Boltzmann) strategy
4. `recommend_article()` - End-to-end recommendation pipeline
5. `get_arm_index()` - Context-category to arm mapping
6. `run_bandit_simulation()` - Simulation runner

### Key Functions

```python
# Helper functions
get_arm_index(user_context_idx, category_idx)  # Maps context+category to arm
get_user_arms(user_context_idx)                 # Returns valid arms for context

# Bandit classes
EpsilonGreedyContextual(n_arms, epsilon)        # ε-greedy algorithm
UCBContextual(n_arms, c)                        # UCB algorithm
SoftMaxContextual(n_arms, tau)                  # SoftMax algorithm

# Simulation
run_bandit_simulation(bandit, sampler, X_test, classifier, T)
```

---

## Conclusions

### Key Findings

1. **UCB algorithm with C=2.0** outperforms other strategies with **8.4585 average reward**
2. **Epsilon-Greedy with low ε (0.01)** provides competitive performance at **8.0218**
3. **User classification** achieves **90% accuracy**, enabling reliable contextualization
4. **Hyperparameter selection is critical**, especially for UCB (C=2.0 vs C=1.0 shows 4.28 reward difference)

### Technical Insights

- **Exploration vs Exploitation:** Higher exploration is beneficial in early stages; UCB's adaptive approach outperforms fixed rates
- **Contextual Learning:** Different user contexts exhibit different reward distributions, justifying CMAB
- **Convergence:** UCB converges faster; Epsilon-Greedy provides stability

### Recommendations for Production

1. **Use UCB with C=2.0** as the primary algorithm
2. Implement **decaying exploration** for Epsilon-Greedy if used
3. Monitor and **retrain user classifier** periodically as user behavior evolves
4. Consider **hybrid approaches** combining multiple strategies
5. Perform **A/B testing** before full deployment

---

## How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
# Install rlcmab_sampler (provided in assignment)
```

### Execution Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   git checkout sohan_U20230162
   ```

2. **Prepare data:**
   - Place `news_articles.csv`, `train_users.csv`, and `test_users.csv` in `data/` folder

3. **Update roll number:**
   - Open `master_final.ipynb`
   - Find the cell with `ROLL_NUMBER = 1`
   - Change to `ROLL_NUMBER = 162`

4. **Run notebook:**
   ```bash
   jupyter notebook master_final.ipynb
   ```
   - Execute all cells sequentially (Kernel → Restart & Run All)

5. **View results:**
   - All plots and metrics will be generated inline
   - Summary statistics displayed at the end

### Expected Runtime

- Data preprocessing: ~30 seconds
- User classification: ~1 minute
- Bandit simulations (9 configurations): ~5-10 minutes
- Total: ~12-15 minutes

---


## Visualizations

The notebook generates the following key visualizations:

1. **Cumulative Rewards Over Time** - Shows total reward accumulation for each algorithm
2. **Moving Average Rewards** - Smoothed reward trends (100-step window)
3. **Arm Pull Distribution** - Which arms were selected most frequently
4. **Learned Q-values** - Expected rewards for each arm
5. **Hyperparameter Sensitivity** - Effect of ε, C, and τ on performance
6. **Algorithm Comparison** - Best configurations head-to-head

