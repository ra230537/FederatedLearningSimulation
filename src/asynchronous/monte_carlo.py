import numpy as np


def get_percentiles_timeout(
    percentile_list,
    num_updates,
    min_connection_time,
    max_connection_time,
    speed_tiers,
):
    n_samples = 1_000_000
    rng = np.random.default_rng(42)
    connection_samples = rng.uniform(
        min_connection_time, max_connection_time, size=n_samples
    )

    proportions = np.array([t[3] for t in speed_tiers], dtype=float)
    proportions /= proportions.sum()
    tier_indices = rng.choice(len(speed_tiers), size=n_samples, p=proportions)

    train_samples = np.empty(n_samples)
    for i, (_, lo, hi, _) in enumerate(speed_tiers):
        mask = tier_indices == i
        n_in = int(mask.sum())
        if n_in:
            train_samples[mask] = rng.uniform(lo, hi, size=n_in)

    sum_samples = connection_samples + train_samples
    timeout = np.percentile(sum_samples, percentile_list) * num_updates
    print(f"Os timeouts são {timeout}")
    return timeout