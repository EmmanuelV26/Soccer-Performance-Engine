# Soccer-Performance-Engine
# 📈 Apple Stock Market Clustering & Prediction

An end-to-end machine learning project that implements **K-Means clustering from scratch** (NumPy only) to discover hidden trading regimes in 40 years of Apple stock data, plus a **logistic regression classifier** to predict next-day price direction.

![Cluster Results](images/apple_enriched_clusters.png)

---

## 🔍 What This Project Does

**1. K-Means Clustering (from scratch)**
- Built the entire K-Means algorithm using only NumPy — no scikit-learn
- Clustered 10,000+ trading days into distinct market regimes
- The algorithm discovered 4 behavioral patterns on its own:

| Cluster | Description | Avg Daily Return | Avg Volatility | Days |
|---------|------------|-----------------|----------------|------|
| Rally Days | Strong uptrend, high momentum | +2.27% | 3.23% | 1,851 |
| Calm / Steady | Normal trading, low volatility | +0.10% | 2.01% | 6,065 |
| Recovery Bounce | Bouncing back from dips | +1.31% | 4.88% | 676 |
| Selloff Days | Sharp declines, high volume | -3.43% | 2.96% | 1,325 |

**2. Feature Engineering**
- Engineered 11 new features from raw price/volume data:
  - Daily return, intraday range, overnight gaps
  - Moving averages (5, 20, 50-day)
  - Price relative to trend (close vs MA20, close vs MA50)
  - Rolling volatility (5-day, 20-day)
  - Volume spike indicator (volume vs 20-day average)

**3. Logistic Regression (from scratch)**
- Built a binary classifier from scratch using NumPy
- Predicts whether tomorrow's close price goes up or down
- Result: ~52% accuracy — essentially random, which is actually an important insight about market efficiency

**4. Interactive Dashboard**
- Built an HTML/Chart.js dashboard to explore clusters visually
- Toggle between price, return, volatility, and volume views
- Click to highlight individual clusters

![Interactive Dashboard](images/kmeans_elbow.png)

---

## 🛠️ Tech Stack

- **Python** — core language
- **NumPy** — all algorithms implemented from scratch
- **Pandas** — data loading and manipulation
- **Matplotlib** — static visualizations
- **Chart.js** — interactive HTML dashboard

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy pandas matplotlib openpyxl
```

### Run the from-scratch K-Means demo
```bash
python kmeans_from_scratch.py
```
This generates synthetic data, clusters it, and produces visualizations including an elbow plot.

### Run the full Apple stock analysis
```bash
python apple_analysis.py
```
This loads `data/Apple.xlsx`, engineers features, runs K-Means and logistic regression, and saves all plots.

### Open the interactive dashboard
Open `apple_kmeans_explorer.html` in any browser.

### Use in Jupyter Notebook
```python
from kmeans_from_scratch import KMeans
import pandas as pd
import numpy as np

df = pd.read_excel("data/Apple.xlsx")
# ... see notebook for full walkthrough
```

---

##  Project Structure

```
ml-stock-clustering/
├── README.md
├── kmeans_from_scratch.py       # K-Means algorithm (NumPy only)
├── apple_analysis.py            # Full analysis with feature engineering
├── apple_kmeans_explorer.html   # Interactive dashboard
├── data/
│   └── Apple.xlsx               # 40 years of Apple stock data
└── images/
    ├── kmeans_result.png
    ├── kmeans_elbow.png
    ├── apple_enriched_clusters.png
    ├── apple_enriched_elbow.png
    └── apple_logistic_regression.png
```

---

##  What I Learned

- **Implementing from scratch matters.** Writing K-Means with just NumPy forced me to understand every step — initialization, assignment, centroid updates, convergence. Calling `sklearn.KMeans()` hides all of that.

- **Normalization is critical.** Without normalizing, volume (in the billions) completely dominated the distance calculation and the clusters were meaningless. After normalizing, the algorithm found genuinely useful patterns.

- **Feature engineering changes everything.** Clustering on raw price just separates time periods. Clustering on behavioral features (return, volatility, volume spikes) finds actual trading regimes that repeat throughout history.

- **Some problems are inherently hard.** The logistic regression couldn't beat random chance at predicting stock direction. This isn't a failure — it's a lesson about the efficient market hypothesis. Day-to-day stock movement is essentially random noise.

- **The elbow method works.** The inertia plot showed a clear bend at k=3 for synthetic data and k=4 for the real stock data, matching the natural structure.

---

##  Contact

Built by **Manny** — Business Administration (MIS) student at Washington State University.

Feel free to connect on [LinkedIn](https://linkedin.com/in/YOUR-PROFILE) or reach out with questions!
