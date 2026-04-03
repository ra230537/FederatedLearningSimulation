import numpy as np
from scipy.stats import uniform


def get_percentiles_timeout(
    percentile_list,
    min_connection_time,
    max_connection_time,
    min_train_time,
    max_train_time,
):
    connection_dist = uniform(
        loc=min_connection_time, scale=max_connection_time - min_connection_time
    )
    train_dist = uniform(loc=min_train_time, scale=max_train_time - min_train_time)
    connection_samples = connection_dist.rvs(1_000_000)
    train_samples = train_dist.rvs(1_000_000)
    sum_samples = connection_samples + train_samples
    # In Sync scenario, the timeout is defined per round, so we don't multiply by num_updates
    timeout = np.percentile(sum_samples, percentile_list)
    print(f"Os timeouts são {timeout}")
    return timeout