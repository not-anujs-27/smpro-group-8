
# Irrigation Event Advisory System: Statistical Modeling & Risk Analysis

## Problem Statement
Conventional irrigation infrastructure often relies on fixed timer schedules, leading to over-watering during low-evapotranspiration periods or delayed intervention during rapid soil moisture depletion. This project constructs an empirical, data-driven irrigation advisory model using statistical distributions (Geometric, Binomial, and Poisson). By processing microclimatic telemetry, the system calculates base event probabilities, models operational waiting times, projects multi-day demand capacity risks, and presents actionable recommendations via an interactive Streamlit application.

---

## Dataset Headers
The underlying dataset (`dataset.csv`) contains 30 daily microclimate observations across five parameters:

| Header | Data Type | Physical Description |
| :--- | :--- | :--- |
| `year` | Integer | Observation calendar year (2023) |
| `date_of_year` | Integer | Julian day of the year (Day 91 through 120) |
| `root_zone_soil_wetness` | Float | Normalized soil moisture fraction at the root level [0.0 - 1.0] |
| `max_temperature` | Float | Maximum ambient daytime temperature (°C) |
| `precipitation` | Float | Total daily precipitation accumulation (mm) |

---

## Mathematical Foundation

### 1. Empirical Event Derivation
An irrigation event is defined deterministically as a binary indicator $E_t \in \{0, 1\}$ based on crop stress thresholds:
$$E_t = \mathbb{I}\left(\text{max\_temperature}_t > \tau_{\text{temp}} \land \text{precipitation}_t < \tau_{\text{rain}}\right)$$

Across an observation record of $N$ days, the baseline daily probability $p$ is derived as:
$$p = \frac{1}{N} \sum_{t=1}^{N} E_t$$

### 2. Geometric Distribution (Waiting-Time Modeling)
Models the number of consecutive days $Z$ until the first required irrigation trigger:
$$P(Z = k) = (1 - p)^{k - 1} p, \quad k \in \{1, 2, 3, \dots\}$$
* **Expected Waiting Time:** $\mathbb{E}[Z] = \frac{1}{p}$
* **Variance:** $\text{Var}(Z) = \frac{1 - p}{p^2}$

### 3. Binomial Distribution (Finite Capacity Sizing)
Models the total count of required irrigation events $X$ over a fixed planning horizon of $n$ days:
$$P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}, \quad k \in \{0, 1, \dots, n\}$$
* **Expected Demand:** $\mathbb{E}[X] = n p$
* **Variance:** $\text{Var}(X) = n p (1 - p)$
* **Overload Risk:** $P(X > k_{\text{cap}}) = 1 - \sum_{i=0}^{k_{\text{cap}}} P(X = i)$

### 4. Poisson Distribution (Event Arrival Approximation)
Approximates the probability of observing $y$ independent irrigation demands over period $n$, parameterized by arrival rate $\lambda = n p$:
$$P(Y = y) = \frac{\lambda^y e^{-\lambda}}{y!}, \quad y \in \{0, 1, 2, \dots\}$$
* **Expected Arrivals:** $\mathbb{E}[Y] = \lambda$
* **Variance:** $\text{Var}(Y) = \lambda$

---

## Calculation and Usage

The project is modularized into three calculation scripts and one visualization dashboard:

* **`geometric_distribution.py`**: Invokes `calculate_geometric_irrigation_wait(p, max_days)` to compute daily survival probabilities, hazard progression, expected waiting interval, and distribution variance[cite: 1].
* **`binomial_distribution.py`**: Evaluates single-point and cumulative probabilities via `binomial(n, p, x)`, outputting exact PMF mass, mean, variance, and standard deviation for horizon sizing[cite: 3].
* **`poisson_distribution.py`**: Evaluates `poisson_probability(lam, y)` to calculate discrete arrival probabilities given average event intensity $\lambda$[cite: 2].
* **`app.py`**: Reads `dataset.csv`, computes $p$ dynamically via user threshold sliders, and visualizes the mathematical models alongside empirical correlation heatmaps and operational alerts.

---

## Inference from Data
Evaluating the 30-day sample under standard stress criteria ($\tau_{\text{temp}} = 38.0^\circ\text{C}$, $\tau_{\text{rain}} = 0.1\text{ mm}$):

* **Baseline Risk:** 11 of 30 days trigger an event, establishing an empirical baseline probability of $p = 0.3667$.
* **Inter-Event Lead Time:** The geometric expectation is $\mathbb{E}[Z] = 2.73\text{ days}$ (variance: $4.71$)[cite: 1]. Preventive line maintenance and reservoir inspections must occur within a 2-day window of dry heat to avoid operational failure.
* **Weekly Capacity Sizing ($n = 7\text{ days}$):** The expected demand is $2.57\text{ days}$[cite: 3]. Peak demand centers on 2 days ($28.77\%$) and 3 days ($27.76\%$)[cite: 3]. The system exhibits a negligible $0.09\%$ risk of continuous 7-day pumping[cite: 3]. Setting pump infrastructure capacity to $k_{\text{cap}} = 4\text{ days}$ limits the overload risk to $6.75\%$[cite: 3].
* **Model Convergence:** At $n = 7$ and $p = 0.3667$, Poisson ($\lambda = 2.57$) moderately diverges from Binomial at the tails ($P_{\text{Bin}}(0) = 4.09\%$ vs. $P_{\text{Pois}}(0) = 7.68\%$)[cite: 2, 3], demonstrating that small planning horizons require the Binomial model for precise risk bounds.

---

## How to Run

### 1. Environment Setup
Create and activate an isolated virtual environment:
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Dependency Installation

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### 3. Launch Application

Start the interactive Streamlit dashboard:

```bash
streamlit run app.py
```