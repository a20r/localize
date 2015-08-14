
import point
import matplotlib.pyplot as plt
import numpy as np


def generate_error_function(anchors, dists):
    def ef(x, y):
        p = point.Point(x, y)
        err = 0
        for anchor, dist in zip(anchors, dists):
            err += abs(dist - p.dist_to(anchor))
        return err
    return ef


def localize(anchors, dists):
    ef = generate_error_function(anchors, dists)
    zs = list()
    xs = np.arange(-1.1, 1.1, 0.01)
    ys = np.arange(-1.1, 1.1, 0.01)
    X, Y = np.meshgrid(xs, ys)
    for x_i, y_i in zip(np.ravel(X), np.ravel(Y)):
        zs.append(ef(x_i, y_i))
    Z = np.array(zs).reshape(X.shape)
    plt.pcolormesh(X, Y, Z)
    plt.show()


if __name__ == "__main__":
    anchors = list()
    anchors.append(point.Point(-1, 0))
    anchors.append(point.Point(1, 0))
    anchors.append(point.Point(0, -1))
    anchors.append(point.Point(0, 1))
    dists = [1, 1, 1, 1]
    localize(anchors, dists)
