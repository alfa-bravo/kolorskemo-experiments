from functools import reduce
from scipy.stats import multivariate_normal
import numpy as np


def get_distribution(vectors, resolution, spread=0.05):
    r = resolution - 1
    grid = np.mgrid[0:resolution, 0:resolution, 0:resolution].T.reshape(-1, 3)
    normalized_points = grid / r
    cov = np.identity(3) * spread / resolution

    def get_distribution_single(v):
        rv = multivariate_normal(v, cov)
        return rv.pdf(normalized_points)

    values = reduce(lambda s, c: s + get_distribution_single(c),
                    vectors,
                    np.zeros(normalized_points.shape[0]))

    output_shape = (resolution, resolution, resolution)
    a = np.zeros(output_shape)
    a[(*grid.T,)] = values
    return a


vectors = [(0.1, 0.5, 0.2)]
values = get_distribution(vectors, 5)
print(values)