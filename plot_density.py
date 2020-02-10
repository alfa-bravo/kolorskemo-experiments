import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from functools import reduce


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
    return grid, a


def show_vectors(vectors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cdict = {
        'red': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        'green': [[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
        'blue': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        'alpha': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    }
    cmap = LinearSegmentedColormap('foo',segmentdata=cdict)
    cmap2 = ListedColormap([(0.0, 0.5, 0.0, 0.0), (0.0, 0.5, 0.0, 1.0)])
    points, values = get_distribution(vectors, 11)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=values[(*points.T,)], cmap=cmap)
    ax.margins(0, 0, 0)
    plt.show()


show_vectors([(0.2, 0.2, 0.2), (0.5, 0.3, 0.8)])