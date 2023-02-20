import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# from skimage.transform import resize
# from skimage.transform import warp, ProjectiveTransform
# from stereo_utils import *
from skimage.color import rgb2gray, rgba2rgb


def points_normalize(points1, points2):
    # validate points
    assert points1.shape[0] == points2.shape[0], "no. of points don't match"

    # compute centroid of points
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)

    # compute the scaling factor
    s1 = np.sqrt(2 / np.mean(np.sum((points1 - c1) ** 2, axis=1)))
    s2 = np.sqrt(2 / np.mean(np.sum((points2 - c2) ** 2, axis=1)))

    # compute the normalization matrix for both the points
    T1 = np.array([[s1, 0, -s1 * c1[0]], [0, s1, -s1 * c1[1]], [0, 0, 1]])
    T2 = np.array([[s2, 0, -s2 * c2[0]], [0, s2, -s2 * c2[1]], [0, 0, 1]])

    # normalize the points
    points1_n = T1 @ points1.T
    points2_n = T2 @ points2.T

    return points1_n, points2_n, T1, T2


def CalcFundamentlMatrix(pts1, pts2):
    pts1, pts2, T1, T2 = points_normalize(pts1, pts2)
    pts1 = pts1.T
    pts2 = pts2.T
    num_points = pts1.shape[0]
    xi = np.zeros((num_points, 9))
    for i in range(num_points):
        xi[i, 0] = pts1[i, 0] * pts2[i, 0]
        xi[i, 1] = pts1[i, 1] * pts2[i, 0]
        xi[i, 2] = pts1[i, 2] * pts2[i, 0]

        xi[i, 3] = pts1[i, 0] * pts2[i, 1]
        xi[i, 4] = pts1[i, 1] * pts2[i, 1]
        xi[i, 5] = pts1[i, 2] * pts2[i, 1]

        xi[i, 6] = pts1[i, 0] * pts2[i, 2]
        xi[i, 7] = pts1[i, 1] * pts2[i, 2]
        xi[i, 8] = pts1[i, 2] * pts2[i, 2]
    U, S, V = np.linalg.svd(xi, full_matrices=True)
    f = V[-1, :]
    F = f.reshape(3, 3)  # reshape f as a matrix

    U, S, V = np.linalg.svd(F, full_matrices=True)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return T2.T @ F @ T1


def plot_line(coeffs, xlim):
    """
    Given the coefficients a, b, c of the ax + by + c = 0,
    plot the line within the given x limits.
    ax + by + c = 0 => y = (-ax - c) / b
    """
    a, b, c = coeffs
    x = np.linspace(xlim[0], xlim[1], 100)
    y = (a * x + c) / -b
    return x, y


def plot_epipolar_lines(img1, img2, points1, points2, show_epipole=False):
    """
    Given two images and their corresponding points, compute the fundamental matrix
    and plot epipole and epipolar lines

    Parameters
    ------------
    img1, img2 - array with shape (height, width)
        grayscale images with only two channels
    points1, points2 - array with shape [n, 3]
        corresponding points in images represented as
        homogeneous coordinates
    show_epipole - boolean
        whether to compute and plot the epipole or not
    """

    # get image size
    h, w = img1.shape
    n = points1.shape[0]
    # validate points
    if points2.shape[0] != n:
        raise ValueError("No. of points don't match")

    # compute the fundamental matrix
    # F = compute_fundamental_matrix_normalized(points1, points2)
    F = CalcFundamentlMatrix(points1, points2)

    # configure figure
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 8))

    # plot image 1
    ax1 = axes[0]
    ax1.set_title("Image 1")
    ax1.imshow(img1, cmap="gray")

    # plot image 2
    ax2 = axes[1]
    ax2.set_title("Image 2")
    ax2.imshow(img2, cmap="gray")

    # plot epipolar lines
    for i in range(n):
        p1 = points1.T[:, i]
        p2 = points2.T[:, i]

        # Epipolar line in the image of camera 1 given the points in the image of camera 2
        coeffs = p2.T @ F
        x, y = plot_line(
            coeffs, (-1500, w)
        )  # limit hardcoded for this image. please change
        ax1.plot(x, y, color="orange")
        ax1.scatter(*p1.reshape(-1)[:2], color="blue")

        # Epipolar line in the image of camera 2 given the points in the image of camera 1
        coeffs = F @ p1
        x, y = plot_line(
            coeffs, (0, 2800)
        )  # limit hardcoded for this image. please change
        ax2.plot(x, y, color="orange")
        ax2.scatter(*p2.reshape(-1)[:2], color="blue")

    if show_epipole:
        # compute epipole
        e1 = compute_epipole(F)
        e2 = compute_epipole(F.T)
        # plot epipole
        ax1.scatter(*e1.reshape(-1)[:2], color="red")
        ax2.scatter(*e2.reshape(-1)[:2], color="red")
    else:
        # set axes limits
        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)
        ax2.set_xlim(0, w)
        ax2.set_ylim(h, 0)

    plt.tight_layout()
    plt.show()


def compute_epipole(F):
    """
    Compute epipole using the fundamental matrix.
    pass F.T as argument to compute the other epipole
    """
    U, S, V = np.linalg.svd(F)
    e = V[-1, :]
    e = e / e[2]
    return e


def main():
    # load images
    im1 = io.imread("data/bench/right.png")
    im1 = rgb2gray(rgba2rgb(im1))
    im2 = io.imread("data/bench/left.png")
    im2 = rgb2gray(rgba2rgb(im2))

    # load matching points
    points1 = np.load("data/bench/right_points.npy")
    points2 = np.load("data/bench/left_points.npy")

    assert points1.shape == points2.shape

    # Fundamental matrix
    F = CalcFundamentlMatrix(points1, points2)
    plot_epipolar_lines(im1, im2, points1, points2, show_epipole=False)


if __name__ == "__main__":
    main()
