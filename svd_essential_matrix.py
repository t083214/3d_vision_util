import numpy as np
import cv2 as cv


# world to image corrdinate (f=1)
def world_to_image(Xw, f, c):
    num_points = Xw.shape[0]
    xc = np.zeros((num_points, 3))
    for i in range(num_points):
        xc[i, :] = Xw[i, :] / Xw[i, 2] * f / c
    return xc


def compute_F_mat(xc, xc_tr):

    num_points = xc.shape[0]
    xi = np.zeros((num_points, 9))

    for i in range(num_points):
        xi[i, 0] = xc[i, 0] * xc_tr[i, 0]
        xi[i, 1] = xc[i, 0] * xc_tr[i, 1]
        xi[i, 2] = xc[i, 0]

        xi[i, 3] = xc[i, 1] * xc_tr[i, 0]
        xi[i, 4] = xc[i, 1] * xc_tr[i, 1]
        xi[i, 5] = xc[i, 1]

        xi[i, 6] = xc_tr[i, 0]
        xi[i, 7] = xc_tr[i, 1]
        xi[i, 8] = 1.0

    e_vals, e_vecs = np.linalg.eig(np.dot(xi.T, xi))
    # Extract the eigenvector (column) associated with the minimum eigenvalue
    min_eig_vec = e_vecs[:, np.argmin(e_vals)]
    min_eig_vec = min_eig_vec.real

    # print("min eigen vector")
    # print(min_eig_vec)

    # for i in range(num_points):
    #     print(np.dot(xi[i, :], min_eig_vec))

    F_ = min_eig_vec.reshape([3, 3])

    return min_eig_vec, F_


def F_svd_and_rank_reduce(F_):
    # rank 3 -> 2
    U, s, V = np.linalg.svd(F_, full_matrices=True)
    print("U : ")
    print(U)
    print(U.shape)

    print("s : ")
    print(s)
    print(s.shape)

    print("V : ")
    print(V)
    print(V.shape)

    s_rank2 = np.eye(3) * s
    s_rank2[2, 2] = 0.0

    F_rank2 = np.dot(np.dot(U, s_rank2), V.T)
    print(F_rank2)

    return F_rank2, U, s_rank2, V


def compute_s1_r1(U, Z, W, V):

    s1 = np.dot(np.dot(U, Z), U.T)
    r1 = np.dot(np.dot(U, np.linalg.inv(W)), V.T)

    return s1, r1


def compute_s2_r2(U, Z, W, V):
    s2 = np.dot(np.dot(U, Z.T), U.T)
    r2 = -1.0 * np.dot(np.dot(U, W), V.T)

    return s2, r2


def get_rotation_translation(F_):
    F_rank2, U, s_rank2, V = F_svd_and_rank_reduce(F_)

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

    s1, r1 = compute_s1_r1(U, Z, W, V)
    s2, r2 = compute_s2_r2(U, Z, W, V)

    print("s1")
    print(s1)
    print("r1")
    print(r1)

    print("s2")
    print(s2)
    print("r2")
    print(r2)

    rotation_ = r1
    translation_ = s1

    return rotation_, translation_


# sample point data
num_points = 9
Xw = np.zeros((num_points, 3))
for i in range(num_points):
    Xw[i, 0] = -0.4 + 0.1 * i
    Xw[i, 1] = 0.5
    Xw[i, 2] = 5

xc = world_to_image(Xw, 1.0, 1.0)

# translate Xw
Xw_tr = Xw
for i in range(Xw.shape[0]):
    Xw_tr[i, 0] = Xw[i, 0] - 0.4

xc_tr = world_to_image(Xw_tr, 1.0, 1.0)

xc = xc * 100 + 320
xc_tr = xc_tr * 100 + 320

print("xc")
print(xc)
print("xc_tr")
print(xc_tr)

print("Xw")
print(Xw)
print("Xw_tr")
print(Xw_tr)


_, F_ = compute_F_mat(xc, xc_tr)

print("F (rank is still not 2)")
print(F_)
rotation_, translation_ = get_rotation_translation(F_)

# opencv reference
pts1 = []
pts2 = []

for i in range(xc.shape[0]):

    pt_ = tuple([int(xc[i, 0]), int(xc[i, 1])])
    pts1.append(pt_)
    pt_ = tuple([int(xc_tr[i, 0]), int(xc_tr[i, 1])])
    pts2.append(pt_)


F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
print(F)

# print(coef_mat.shape)
# print(coef_mat)

# eig = np.linalg.eig(coef_mat)
# M = np.dot(coef_mat, coef_mat.T) * 1 / num_points

# Y = np.zeros((9, 1))
# # LeastSquare(coef_mat, Y)
# # print(Fundamental)
