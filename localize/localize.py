
import point
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import random


def simulate_dists(anchors, source, sigma, num_samples=1):
    if type(sigma) == float or type(sigma) == int:
        sigmas = np.array([sigma for _ in xrange(len(anchors))])
    dists = np.zeros((len(anchors), num_samples))
    for i, anchor in enumerate(anchors):
        r_dist = anchor.dist_to(source)
        dists[i] = r_dist + random.gauss(0, sigmas[i])
    return dists


def generate_error_function(anchors, dists):
    def ef(x, y=None, z=0):
        if y is None:
            p = point.Point(x[0], x[1], x[2])
        else:
            p = point.Point(x, y, z)
        err = 0
        for anchor, dist in zip(anchors, dists):
            err += abs(dist - p.dist_to(anchor))
        return err
    return ef


def get_bounds(anchors, dists):
    x_min, x_max, y_min, y_max = None, None, None, None
    for a, d in zip(anchors, dists):
        if x_min is None or a.x - d < x_min:
            x_min = a.x - d
        if y_min is None or a.y - d < y_min:
            y_min = a.y - d
        if x_max is None or a.x + d > x_max:
            x_max = a.x + d
        if y_max is None or a.y + d > y_max:
            y_max = a.y + d
    return x_min, x_max, y_min, y_max


def locations(anchors, dists):
    if dists.shape[1] == 1:
        ef = generate_error_function(anchors, dists)
        loc = opt.minimize(ef, anchors[0].to_np_array())
        return point.Point(loc.x[0], loc.x[1], loc.x[2])
    else:
        for i in xrange(dists.shape[0]):
            pass


def plot_error(anchors, dists, pl, x_step=0.05, y_step=0.05):
    ef = generate_error_function(anchors, dists)
    zs = list()
    x_min, x_max, y_min, y_max = get_bounds(anchors, dists)
    xs = np.arange(x_min, x_max, x_step)
    ys = np.arange(y_min, y_max, y_step)
    X, Y = np.meshgrid(xs, ys)
    for x_i, y_i in zip(np.ravel(X), np.ravel(Y)):
        zs.append(ef(x_i, y_i))
    Z = np.array(zs).reshape(X.shape)
    pl.pcolormesh(X, Y, Z)
    return plt


if __name__ == "__main__":
    anchors = list()
    anchors.append(point.Point(-1, 0))
    anchors.append(point.Point(1, 0))
    anchors.append(point.Point(0, -1))
    anchors.append(point.Point(0, 1))
    anchors.append(point.Point(0, 2))
    source = point.Point(0, 0)
    sigma = 0.001
    dists = simulate_dists(anchors, source, sigma)
    print locations(anchors, dists)
    plot_error(anchors, dists, plt).show()
