import numpy as np

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hpd(x, alpha=0.1):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)

    """

    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))
    
from tqdm import tqdm
def get_ranges(x, level=90):
    #med, low, hi = np.median(x), np.percentile(x, (100-level)/2),np.percentile(x, 100 - (100-level)/2)
    low, hi = hpd(x, alpha=(100-level)*0.01)
    return low, np.median(x), hi

def print_ranges(low, med, hi, level=90, digits=2):
    return f"{np.round(med, digits)}" + r"^{+" + f"{np.round(hi-med, digits)}" + r"}_{-" + f"{np.round(med-low, digits)}" + r"}"

def get_bayes_factor(samples, variables, limit, a, b, prior_value=1, 
                     quantile_level=0.9, trials=100, bandwidth_scale=1,
                     log_bfs=False):
    print("Importing truncatedgaussianmixtures, install it if you want to use this function")
    from truncatedgaussianmixtures import fit_kde
    bfs = np.zeros(trials);
    for i in tqdm(range(trials)):
        fit = fit_kde(samples[variables].sample(len(samples), replace=True),a,b, bandwidth_scale=bandwidth_scale);
        bfs[i] = fit.pdf(np.asarray(limit))/prior_value

    percentile_range_delta = ((1 - quantile_level)/2)*100
    percentile_ranges = (percentile_range_delta, 100-percentile_range_delta)
        
    if log_bfs:
        log_bfs = np.log10(bfs)
        med, low, hi = np.median(log_bfs), np.percentile(log_bfs, percentile_ranges[0]),np.percentile(log_bfs, percentile_ranges[1])
        #low, hi = hpd(1/bfs, alpha=(100-level)*0.01)
        return (low, med, hi), log_bfs
    else:
        med, low, hi = np.median(bfs), np.percentile(bfs, percentile_ranges[0]),np.percentile(bfs, percentile_ranges[1])
        #low, hi = hpd(1/bfs, alpha=(100-level)*0.01)
        return (low, med, hi), bfs