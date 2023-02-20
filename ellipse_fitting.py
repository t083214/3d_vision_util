# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def make_circle(a=1, b=1, xy=(0, 0), phi=0, n=100, random_s=0.1):
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / n)
    X = a * np.cos(theta)
    Y = b * np.sin(theta)
    data_mat = np.matrix(np.vstack([X, Y]))
    phi_d = np.deg2rad(phi)
    rot = np.matrix([[np.cos(phi_d), -np.sin(phi_d)], [np.sin(phi_d), np.cos(phi_d)]])
    rot_data = rot * data_mat
    X = rot_data[0].A
    Y = rot_data[1].A

    rand1 = np.random.normal(scale=random_s, size=theta.shape)
    rand2 = np.random.normal(scale=random_s, size=theta.shape)

    return X + xy[0], Y + xy[1]


def ellipse_fitting(X, Y):

    num_points = X.shape[0]
    xi = np.zeros((num_points, 6))
    f0 = 1.0

    for i in range(num_points):
        xi[i, 0] = X[0, i] * X[0, i]
        xi[i, 1] = 2.0 * X[0, i] * Y[0, i]
        xi[i, 2] = Y[0, i] * Y[0, i]
        xi[i, 3] = 2.0 * f0 * X[0, i]
        xi[i, 4] = 2.0 * f0 * Y[0, i]
        xi[i, 5] = f0 * f0

    e_vals, e_vecs = np.linalg.eig(np.dot(xi.T, xi))

    # Extract the eigenvector (column) associated with the minimum eigenvalue
    z = e_vecs[:, np.argmin(e_vals)]

    print("hyp")
    print(z)
    a = z

    # -------------------Fit ellipse-------------------
    b, c, d, f, g, a = z[1] / 2.0, z[2], z[3] / 2.0, z[4] / 2.0, z[5], z[0]

    print(b, c, d, f, g, a)

    num = b * b - a * c
    cx = (c * d - b * f) / num
    cy = (a * f - b * d) / num

    angle = 0.5 * np.arctan(2 * b / (a - c)) * 180 / np.pi
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    a = np.sqrt(abs(up / down1))
    b = np.sqrt(abs(up / down2))

    # ---------------------Get path---------------------
    ell = Ellipse((cx, cy), a * 2.0, b * 2.0, angle)
    ell_coord = ell.get_verts()

    params = [cx, cy, a, b, angle]

    return params, ell_coord


def fitEllipse(cont, method):

    x = cont[:, 0]
    y = cont[:, 1]

    x = x[:, None]
    y = y[:, None]

    D = np.hstack([x * x, x * y, y * y, x, y, np.ones(x.shape)])
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))

    if method == 1:
        n = np.argmax(np.abs(E))
    else:
        n = np.argmax(E)
    a = V[:, n]

    # -------------------Fit ellipse-------------------
    b, c, d, f, g, a = a[1] / 2.0, a[2], a[3] / 2.0, a[4] / 2.0, a[5], a[0]
    num = b * b - a * c
    cx = (c * d - b * f) / num
    cy = (a * f - b * d) / num

    angle = 0.5 * np.arctan(2 * b / (a - c)) * 180 / np.pi
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    a = np.sqrt(abs(up / down1))
    b = np.sqrt(abs(up / down2))

    # ---------------------Get path---------------------
    ell = Ellipse((cx, cy), a * 2.0, b * 2.0, angle)
    ell_coord = ell.get_verts()

    params = [cx, cy, a, b, angle]

    return params, ell_coord


def plotConts(contour_list):
    """Plot a list of contours"""

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    for ii, cii in enumerate(contour_list):
        x = cii[:, 0]
        y = cii[:, 1]
        ax2.plot(x, y, "-")
    plt.show()


def main():

    X, Y = make_circle(a=1, b=1, xy=(0, 0), phi=0, n=100, random_s=0.0001)

    # params_hyp, ell2 = ellipse_fitting(X, Y)
    # print(params_hyp)

    data = np.vstack((X, Y)).T
    params1, ell1 = fitEllipse(data, 1)
    print(params1)

    print(type(ell1))
    print(type(data))
    print(data.shape)
    print(ell1.shape)

    plotConts([data, ell1])


if __name__ == "__main__":
    main()
