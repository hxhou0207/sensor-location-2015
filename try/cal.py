import numpy as np


def corrected_jacobian(delta_x, delta_y, delta_z):
    d_xy = delta_x ** 2 + delta_y ** 2
    d_xyz = d_xy + delta_z ** 2
    sqrt_d_xy = np.sqrt(d_xy)

    H = np.zeros((2, 6))

    # 对方位角（azimuth）的偏导数
    H[0, 0] = -delta_y / d_xy
    H[0, 1] = delta_x / d_xy
    H[0, 2] = 0
    H[0, 3] = delta_y / d_xy
    H[0, 4] = -delta_x / d_xy
    H[0, 5] = 0

    # 对仰角（elevation）的偏导数
    H[1, 0] = -delta_x * delta_z / (d_xyz * sqrt_d_xy)
    H[1, 1] = -delta_y * delta_z / (d_xyz * sqrt_d_xy)
    H[1, 2] = sqrt_d_xy / d_xyz
    H[1, 3] = delta_x * delta_z / (d_xyz * sqrt_d_xy)
    H[1, 4] = delta_y * delta_z / (d_xyz * sqrt_d_xy)
    H[1, 5] = -sqrt_d_xy / d_xyz

    return H


def h(X):
    delta_x = X[0] - X[3]
    delta_y = X[1] - X[4]
    delta_z = X[2] - X[5]
    theta = np.arctan2(delta_y, delta_x)
    d_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)
    phi = np.arctan2(delta_z, d_xy)
    return np.array([theta, phi])


def numerical_jacobian(X, epsilon=1e-6):
    n = len(X)
    m = 2  # 2 output dimensions (theta, phi)
    J_numeric = np.zeros((m, n))

    for i in range(n):
        perturb = np.zeros(n)
        perturb[i] = epsilon
        X_plus = X + perturb
        X_minus = X - perturb

        f_plus = h(X_plus)
        f_minus = h(X_minus)

        J_numeric[:, i] = (f_plus - f_minus) / (2 * epsilon)

    return J_numeric


# 示例调用
delta_x = 1.0
delta_y = 2.0
delta_z = 3.0

# 构建状态向量 X = [x_e, y_e, z_e, x_r, y_r, z_r]
X = np.array([delta_x, delta_y, delta_z, 0.0, 0.0, 0.0])

H_corrected = corrected_jacobian(delta_x, delta_y, delta_z)
H_numeric = numerical_jacobian(X)

print("Corrected Jacobian:\n", H_corrected)
print("Numerical Jacobian:\n", H_numeric)
print("Difference between Corrected and Numerical:\n", H_corrected - H_numeric)
