import numpy as np
import corner
import matplotlib.pyplot as plt


def make_faulty_data(n=10000, d=6):
    orig_data = np.random.uniform(low=-1, high=1, size=(n, d))
    ind = np.argwhere(abs(orig_data[:,1]-0.5) < 0.2)[:,0]
    faulty_data = orig_data[ind]
    return orig_data, faulty_data

def get_bounds(data):
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    return [(lo, hi) for lo, hi in zip(mins, maxs)]

def plot_high_dim_data(samples, param_names=None, bounds=None):
    assert len(samples.shape)==2
    assert samples.shape[0] > samples.shape[1]

    if param_names is None:
        param_names = [fr"x$_{i}$" for i in range(samples.shape[1])]

    # Make corner plot
    figure = corner.corner(
        samples,
        labels=param_names,
        show_titles=True,
        range = bounds,
        title_kwargs={"fontsize": 12}
    )

if __name__ == "__main__":
    orig_data, faulty_data = make_faulty_data()
    bounds = get_bounds(orig_data)
    plot_high_dim_data(faulty_data, bounds=bounds)
    plt.show()
