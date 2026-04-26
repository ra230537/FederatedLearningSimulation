def clipped_learning_ratio(
    eta_base: float,
    f_i: float,
    f_mean: float,
    beta: float = 0.5,
    eta_min: float = 0.5,
    eta_max: float = 2.0,
    epsilon: float = 1e-8,
) -> float:
    """
    eta_i = eta_base * clip((f_mean / (f_i + epsilon)) ** beta, eta_min, eta_max)
    """

    frequency_factor = (f_mean / (f_i + epsilon)) ** beta

    clipped_factor = max(eta_min, min(frequency_factor, eta_max))

    eta_i = eta_base * clipped_factor

    return eta_i
