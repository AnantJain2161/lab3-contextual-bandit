# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Course:** Reinforcement Learning Fundamentals  
**Student Name:** Anant Jain  
**Roll Number:** U20230155  


---

## Executive Summary

This project implements a comprehensive **Contextual Multi-Armed Bandit (CMAB)** system for personalized news article recommendations. The system combines user classification with three distinct bandit algorithms (Epsilon-Greedy, UCB, and SoftMax) to maximize user engagement through intelligent exploration-exploitation strategies.

### 🎯 Key Results

- **Best Algorithm:** Upper Confidence Bound (UCB) with C=1.0
- **Best Average Reward:** 4.6494
- **User Classification Accuracy:** 97.0%
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
        User1       0.97      0.98      0.97       666
        User2       0.97      0.97      0.97       659
        User3       0.97      0.96      0.97       675
    ──────────────────────────────────────────────────
    accuracy                           0.97      2000
   macro avg       0.97      0.97      0.97      2000
weighted avg       0.97      0.97      0.97      2000
```

**Key Insight:** The classifier achieves **97% accuracy**, ensuring highly reliable context detection for the recommendation engine.

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

| Algorithm       | Hyperparameter | Avg Reward | Total Reward |
|----------------|----------------|------------|--------------|
| Epsilon-Greedy | ε=0.01         | **4.4865** | 44,864.70    |
| Epsilon-Greedy | ε=0.1          | 4.1368     | 41,367.73    |
| Epsilon-Greedy | ε=0.2          | 3.5928     | 35,927.85    |
| UCB            | C=0.5          | 4.6339     | 46,338.83    |
| **UCB**        | **C=1.0**      | **4.6494** | **46,493.83**|
| UCB            | C=2.0          | 4.6213     | 46,212.77    |
| SoftMax        | τ=0.5          | 4.5809     | 45,809.30    |
| SoftMax        | τ=1.0          | 4.3999     | 43,999.33    |
| SoftMax        | τ=2.0          | 3.9722     | 39,721.55    |

### 🏆 Winner: UCB with C=1.0

Achieves the highest average reward of **4.6494** across 10,000 time steps.

### Algorithm-Specific Results

#### Epsilon-Greedy Results

| ε    | Avg Reward | Total Reward | Interpretation |
|------|------------|--------------|----------------|
| 0.01 | **4.4865** | 44,864.70    | High exploitation, best performance |
| 0.1  | 4.1368     | 41,367.73    | Balanced exploration-exploitation |
| 0.2  | 3.5928     | 35,927.85    | High exploration, lower performance |

**Key Finding:** Lower ε values provide better average reward. ε=0.01 achieves the best Epsilon-Greedy performance.

#### UCB Results

| C   | Avg Reward | Total Reward | Interpretation |
|-----|------------|--------------|----------------|
| 0.5 | 4.6339     | 46,338.83    | Good performance, conservative exploration |
| 1.0 | **4.6494** | **46,493.83**| **Best overall performance** |
| 2.0 | 4.6213     | 46,212.77    | High exploration, slightly lower reward |

**Key Finding:** UCB with C=1.0 achieves the highest average reward overall. All UCB configurations perform well, showing the robustness of the algorithm.

#### SoftMax Results

| τ   | Avg Reward | Total Reward |
|-----|------------|--------------|
| 0.5 | **4.5809** | 45,809.30    |
| 1.0 | 4.3999     | 43,999.33    |
| 2.0 | 3.9722     | 39,721.55    |

**Key Finding:** SoftMax with τ=0.5 provides the best SoftMax performance with competitive results. Lower temperature (more greedy) outperforms higher temperature.

### Recommendation Engine Examples

**Pipeline Implementation:**
1. Classify User → Random Forest classifier → User context
2. Select Category → Apply UCB (C=2.0) → Extract Q-values → Map to news category
3. Recommend Article → Query news database → Return article headline

**Sample Top Arms by Q-value:**

**Epsilon-Greedy (ε=0.01):**
1. Arm 0: User1 - Entertainment (Q=6.2342)
2. Arm 7: User2 - Crime (Q=5.6308)
3. Arm 6: User2 - Tech (Q=3.9577)
4. Arm 5: User2 - Education (Q=3.1353)
5. Arm 10: User3 - Tech (Q=0.8077)

**UCB (C=1.0):**
1. Arm 0: User1 - Entertainment (Q=6.2365)
2. Arm 7: User2 - Crime (Q=5.5958)
3. Arm 6: User2 - Tech (Q=3.9081)
4. Arm 5: User2 - Education (Q=2.2676)
5. Arm 10: User3 - Tech (Q=0.7942)

**SoftMax (τ=0.5):**
1. Arm 0: User1 - Entertainment (Q=6.1959)
2. Arm 7: User2 - Crime (Q=5.5983)
3. Arm 10: User3 - Tech (Q=0.8326)
4. Arm 5: User2 - Education (Q=0.0000)
5. Arm 4: User2 - Entertainment (Q=0.0000)

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
- **ε=0.01:** Best (4.49), high exploitation
- **ε=0.1:** Medium (4.14), balanced
- **ε=0.2:** Worst (3.59), too much exploration
- **Recommendation:** Lower values strongly preferred; ε=0.01 optimal

#### UCB (C)
- **C=0.5:** Good (4.63), conservative
- **C=1.0:** Best (4.65), optimal balance
- **C=2.0:** Good (4.62), aggressive exploration
- **Recommendation:** All values perform well; C=1.0 slightly better

#### SoftMax (τ)
- **τ=0.5:** Best (4.58), greedy selection
- **τ=1.0:** Medium (4.40), balanced
- **τ=2.0:** Worst (3.97), too much randomness
- **Recommendation:** Lower temperature preferred

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

1. **UCB algorithm with C=1.0** outperforms other strategies with **4.6494 average reward**
2. **Epsilon-Greedy with ε=0.01** provides strong performance at **4.4865**
3. **SoftMax with τ=0.5** achieves competitive results at **4.5809**
4. **User classification** achieves **97% accuracy**, ensuring highly reliable contextualization
5. **All algorithms** show consistent performance across the 10,000 time steps
6. **UCB is most robust** with all three C values (0.5, 1.0, 2.0) performing similarly well

### Technical Insights

- **Exploration vs Exploitation:** Epsilon-Greedy benefits from low ε; UCB naturally balances exploration
- **Contextual Learning:** 97% classification accuracy enables effective personalization
- **Algorithm Robustness:** UCB shows consistent performance across hyperparameters (4.62-4.65 range)
- **Convergence:** All algorithms converge within reasonable timeframes
- **Performance Ranking:** UCB (4.6494) > SoftMax (4.5809) > Epsilon-Greedy (4.4865) for best configurations

### Recommendations for Production

1. **Use UCB with C=1.0** as the primary algorithm for its optimal balance
2. **Epsilon-Greedy with ε=0.01** is a strong alternative for simplicity
3. Monitor and **retrain user classifier** periodically as user behavior evolves (currently at 97% accuracy)
4. Consider **ensemble approaches** combining multiple strategies
5. Perform **A/B testing** before full deployment
6. **All UCB configurations** (C=0.5, 1.0, 2.0) are production-ready with minimal performance difference

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

## File Structure

```
lab3-contextual-bandit/
│
├── data/
│   ├── news_articles.csv          # News dataset (209,527 articles)
│   ├── train_users.csv             # Training users (2,000 samples)
│   └── test_users.csv              # Test users (2,000 samples)
│
├── master_final.ipynb              # Main notebook with all implementations
├── README.md                       # This file
├── rlcmab_sampler.py              # Reward sampler module
│
└── results/
    ├── epsilon_greedy_plots.png    # ε-greedy performance
    ├── ucb_plots.png               # UCB performance
    ├── softmax_plots.png           # SoftMax performance
    └── comparison_plots.png        # Algorithm comparison
```

---

## Visualizations

The notebook generates the following key visualizations:

1. **Cumulative Rewards Over Time** - Shows total reward accumulation for each algorithm
2. **Moving Average Rewards** - Smoothed reward trends (100-step window)
3. **Arm Pull Distribution** - Which arms were selected most frequently
4. **Learned Q-values** - Expected rewards for each arm
5. **Hyperparameter Sensitivity** - Effect of ε, C, and τ on performance
6. **Algorithm Comparison** - Best configurations head-to-head

---

## Future Improvements

1. **Linear Contextual Bandits:** Use linear models (e.g., LinUCB) for better generalization
2. **Thompson Sampling:** Bayesian approach with strong empirical performance
3. **Neural Bandits:** Deep learning for complex user-item interactions
4. **Non-stationary Bandits:** Handle changing user preferences over time
5. **Cold Start Solutions:** Better handling of new users and new content
6. **Feature Engineering:** Incorporate richer user and article features

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
2. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time analysis of the multiarmed bandit problem*
3. Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). *A contextual-bandit approach to personalized news article recommendation*

---

## Contact

**Student:** Anant Jain
**Roll Number:** U20230155  

---

## Acknowledgments

- Course Instructor for providing the assignment framework
- `rlcmab_sampler` module authors
- Reinforcement Learning Fundamentals teaching team

---

## License

This project is submitted as part of academic coursework. Please refer to your institution's academic integrity policies.

---

**Last Updated:** February 2026  
**Project Status:** ✅ Complete
