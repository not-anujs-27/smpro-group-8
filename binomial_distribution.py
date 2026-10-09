import math


def binomial(n, p, x):
    """
    Calculate the binomial probability, mean, variance, and standard deviation.

    Parameters:
    - n (int): Number of trials.
    - p (float): Probability of success in a single trial.
    - x (int): Number of successful trials.

    Returns:
    - tuple: (prob, mean, variance, sd)
    """
    # Calculate probability mass using combinations: nCx * p^x * (1-p)^(n-x)
    prob = math.comb(n, x) * p**x * (1 - p) ** (n - x)

    # Calculate statistical properties
    mean = n * p
    variance = n * p * (1 - p)
    sd = math.sqrt(variance)

    return prob, mean, variance, sd


if __name__ == "__main__":
    prob, mean, variance, sd = binomial(10, 0.5, 5)
    print(f"Probability: {prob}, Mean: {mean}, Variance: {variance}, SD: {sd}")
