import math


def poisson_probability(lam, y):
    """
    Calculate the Poisson probability of getting exactly y events.

    Parameters:
    - lam (float): The average number of events (lambda).
    - y (int): The number of events.

    Returns:
    - float: The Poisson probability.
    """
    # Calculate Poisson PMF: (e^-lambda * lambda^y) / y!
    return math.exp(-lam) * lam**y / math.factorial(y)


if __name__ == "__main__":
    prob = poisson_probability(5.0, 2)
    print(f"Poisson probability (lambda=5.0, y=2): {prob}")
