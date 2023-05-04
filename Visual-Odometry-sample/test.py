from Algorithm import VisualOdometry
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
import numpy as np


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
    visual_odometry = VisualOdometry()
    F = visual_odometry.calcualte_fundamental_matrix(points1, points2)
    print(F)
    # fundamental_mat_estimator = FundametanMatrix()
    # Fundamental matrix
    # F = fundamental_mat_estimator.CalcFundamentlMatrix(points1, points2)
    # fundamental_mat_estimator.plot_epipolar_lines(
    #     im1, im2, points1, points2, show_epipole=False
    # )


if __name__ == "__main__":
    main()
