import numpy as np
from scipy.stats import geom
import pandas as pd


def calculate_geometric_irrigation_wait(p: float, max_days: int = 15):
    """
    Calculates the geometric probability distribution for waiting-time
    until the next required irrigation event.

    Parameters:
    - p (float): Probability of soil moisture reaching critical depletion on any given day (0 < p <= 1).
    - max_days (int): Maximum number of consecutive dry-down days (k) to evaluate.

    Returns:
    - dict: Contains probabilities, expected waiting time, variance, and a summary DataFrame.
    """
    if not (0 < p <= 1):
        raise ValueError("Probability p must be strictly between 0 and 1.")

    k_values = np.arange(1, max_days + 1)

    # Calculate Probability Mass Function (PMF) for each day
    probabilities = geom.pmf(k_values, p)

    expected_value = geom.mean(p)  # E[X] = 1/p
    variance = geom.var(p)  # Var[X] = (1-p)/p^2

    df_summary = pd.DataFrame(
        {
            "Consecutive_Days_k": k_values,
            "Probability_Pk": probabilities,
            "Cumulative_Probability": np.cumsum(probabilities),
        }
    )

    results = {
        "parameter_p": p,
        "expected_wait_days": float(expected_value),
        "variance": float(variance),
        "summary_table": df_summary,
    }

    return results


if __name__ == "__main__":
    analysis = calculate_geometric_irrigation_wait(0.25, 10)
    print(f"Expected Waiting Time: {analysis['expected_wait_days']} days")
    print(f"Variance: {analysis['variance']}")
