import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import tensorflow as tf
import io

# 初始化参数
target_position = np.array([1000, 1000, 1000], dtype=np.float64)  # 静态目标位置
sigma_u = np.float64(1)  # 测量噪声的单位距离误差
gamma = np.float64(0.2)  # 功率衰减指数

# 定义日志目录
log_dir = "logs/ekf"
file_writer = tf.summary.create_file_writer(log_dir)

# 状态转移矩阵 F_k
T = np.float64(2)  # 时间间隔
F = np.array([
    [1, T, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, T, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, T],
    [0, 0, 0, 0, 0, 1]
], dtype=np.float64)

# 这里是：将预测位置与sensor位置的关系转化成方位角与仰角
# EKF 预测和更新步骤
def h(X, sensor_position):
    delta_x = X[0] - sensor_position[0]
    delta_y = X[2] - sensor_position[1]
    delta_z = X[4] - sensor_position[2]
    azimuth = np.arctan2(delta_y, delta_x)
    elevation = np.arctan2(delta_z, np.sqrt(delta_x ** 2 + delta_y ** 2))
    return np.array([azimuth, elevation], dtype=np.float64)

# 这里是h的雅可比矩阵，K和P的运算会用到
def H_jacobian(X, sensor_position):
    delta_x = X[0] - sensor_position[0]
    delta_y = X[2] - sensor_position[1]
    delta_z = X[4] - sensor_position[2]
    d_xy = delta_x ** 2 + delta_y ** 2
    d_xyz = d_xy + delta_z ** 2
    sqrt_d_xy = np.sqrt(d_xy)

    H = np.zeros((2, 6), dtype=np.float64)
    if d_xy != 0:
        H[0, 0] = -delta_y / d_xy
        H[0, 2] = delta_x / d_xy
    if d_xyz != 0 and sqrt_d_xy != 0:
        H[1, 0] = -delta_x * delta_z / (d_xyz * sqrt_d_xy)
        H[1, 2] = -delta_y * delta_z / (d_xyz * sqrt_d_xy)
        H[1, 4] = sqrt_d_xy / d_xyz

    return H

def predict(X, P, F, Q):
    X = F @ X
    P = F @ P @ F.T + Q
    return X, P

def update(X, P, z_noisy, H, R, sensor_position):
    y = z_noisy - h(X, sensor_position)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    X = X + K @ y
    P = (np.eye(6, dtype=np.float64) - K @ H) @ P
    return X, P, K, S

def measurement(X, sensor_position, R):
    M = np.array([X[0], 0, X[1], 0, X[2], 0], dtype=np.float64)
    true_measurement = h(M, sensor_position)
    noise = np.random.multivariate_normal([0, 0], R)
    noisy_measurement = true_measurement + noise
    return noisy_measurement

def log_matrix(writer, tag, matrix, step):
    fig, ax = plt.subplots()
    ax.imshow(np.ones_like(matrix), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    with writer.as_default():
        tf.summary.image(tag, image, step=step)

def simulate_ekf(N, Q_val, iterations, radius, height):
    sensor_positions = np.zeros((N, 3), dtype=np.float64)
    for k in range(N):
        theta_k = 2 * np.pi * k / N
        sensor_positions[k, 0] = target_position[0] + radius * np.cos(theta_k)
        sensor_positions[k, 1] = target_position[1] + radius * np.sin(theta_k)
        sensor_positions[k, 2] = height

    X = np.array([1200, 0, 800, 0, 1400, 0], dtype=np.float64)
    P = np.diag([100 ** 2, 0, 100 ** 2, 0, 100 ** 2, 0]).astype(np.float64)
    Q = np.eye(6, dtype=np.float64) * Q_val
    mse_list = []

    sensor1_az_el = None
    sensor50_az_el = None

    for i in range(iterations):
        for k in range(N):
            X, P = predict(X, P, F, Q)
            sensor_position = sensor_positions[k]
            d_k = np.linalg.norm(X[[0, 2, 4]] - sensor_position)
            R_k = np.diag([sigma_u ** 2 * d_k ** gamma, sigma_u ** 2 * d_k ** gamma]).astype(np.float64)
            z_noisy = measurement(np.concatenate([target_position, np.zeros(3)], dtype=np.float64), sensor_position, R_k)
            H = H_jacobian(X, sensor_position)
            X, P, K, S = update(X, P, z_noisy, H, R_k, sensor_position)

            if k == 0:
                sensor1_az_el = np.degrees(z_noisy)
                sensor1_position = sensor_position
            elif k == 49 and N >= 50:
                sensor50_az_el = np.degrees(z_noisy)
                sensor50_position = sensor_position

        mse = np.trace(P)
        mse_list.append(mse)

        log_matrix(file_writer, "X_matrix", X.reshape(-1, 1), i)
        log_matrix(file_writer, "P_matrix", P, i)
        log_matrix(file_writer, "K_matrix", K, i)
        log_matrix(file_writer, "H_matrix", H, i)
        log_matrix(file_writer, "R_matrix", R_k, i)
        log_matrix(file_writer, "S_matrix", S, i)
        log_matrix(file_writer, "h_matrix", z_noisy.reshape(-1, 1), i)

    print(f"MSE after {iterations} iterations: {mse_list[-1]}")
    return sensor_positions, mse_list, sensor1_az_el, sensor50_az_el, sensor1_position, sensor50_position

fig = plt.figure(figsize=(15, 10))
ax_3d = fig.add_subplot(121, projection='3d')
ax_mse = fig.add_subplot(122)
plt.subplots_adjust(left=0.25, bottom=0.35)

ax_sensors = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_Q = plt.axes([0.1, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_radius = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_height = plt.axes([0.1, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')

sensors_slider = Slider(ax_sensors, 'Sensors', 10, 200, valinit=100, valstep=1)
Q_slider = Slider(ax_Q, 'Process Noise (Q)', 0, 1.0, valinit=0, valstep=0.1)
radius_slider = Slider(ax_radius, 'Radius', 4, 2000, valinit=591.80, valstep=1)
height_slider = Slider(ax_height, 'Height', 0, 2000, valinit=462.38, valstep=1)

ax_sensor1_az_el = plt.axes([0.75, 0.95, 0.2, 0.03], facecolor='lightgoldenrodyellow')
ax_sensor50_az_el = plt.axes([0.75, 0.90, 0.2, 0.03], facecolor='lightgoldenrodyellow')

sensor1_az_el_text = TextBox(ax_sensor1_az_el, 'Sensor 1 (az, el)', initial="")
sensor50_az_el_text = TextBox(ax_sensor50_az_el, 'Sensor 50 (az, el)', initial="")

sensor_history = []
mse_history = []

def update_sliders(val):
    N = int(sensors_slider.val)
    Q_val = Q_slider.val
    radius = radius_slider.val
    height = height_slider.val

    sensor_positions, mse_list, sensor1_az_el, sensor50_az_el, sensor1_position, sensor50_position = simulate_ekf(N, Q_val, 1000, radius, height)

    sensor_history.append(sensor_positions)
    mse_history.append(mse_list)

    ax_3d.clear()
    ax_mse.clear()

    colors = plt.cm.viridis(np.linspace(0, 1, len(sensor_history)))
    for i, sensors in enumerate(sensor_history):
        ax_3d.scatter(sensors[:, 0], sensors[:, 1], sensors[:, 2], color=colors[i], label=f'State {i + 1}', alpha=0.6)

    ax_3d.scatter(target_position[0], target_position[1], target_position[2], marker='*', label='True Target Position', color='green', s=200)
    ax_3d.scatter(sensor1_position[0], sensor1_position[1], sensor1_position[2], marker='*', label='Sensor 1', color='red', s=200)
    ax_3d.scatter(sensor50_position[0], sensor50_position[1], sensor50_position[2], marker='*', label='Sensor 50', color='blue', s=200)

    ax_3d.text(sensor1_position[0], sensor1_position[1], sensor1_position[2], f"{sensor1_position[0]:.1f}, {sensor1_position[1]:.1f}, {sensor1_position[2]:.1f}", color='red')
    ax_3d.text(sensor50_position[0], sensor50_position[1], sensor50_position[2], f"{sensor50_position[0]:.1f}, {sensor50_position[1]:.1f}, {sensor50_position[2]:.1f}", color='blue')

    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend()

    for i, mse_list in enumerate(mse_history):
        if any(mse_list):
            ax_mse.plot(range(1, 101), mse_list, color=colors[i], label=f'State {i + 1} MSE')

    all_mse = np.concatenate(mse_history)
    if len(all_mse) > 0:
        min_mse = all_mse.min()
        max_mse = all_mse.max()
        min_mse = max(min_mse, 1e-10)
        ax_mse.set_ylim([min_mse * 0.9, max_mse * 1.1])
        ax_mse.set_yscale('log')

    ax_mse.set_xlabel('Iteration')
    ax_mse.set_ylabel('MSE')
    ax_mse.legend()

    ax_mse.text(0.5, 0.1, f"Start MSE: {mse_list[0]:.2f}", transform=ax_mse.transAxes, ha='center', va='center', color='black')
    ax_mse.text(0.5, 0.9, f"End MSE: {mse_list[-1]:.2f}", transform=ax_mse.transAxes, ha='center', va='center', color='black')

    sensor1_az_el_text.set_val(f"az: {sensor1_az_el[0]:.2f}, el: {sensor1_az_el[1]:.2f}")
    if sensor50_az_el is not None:
        sensor50_az_el_text.set_val(f"az: {sensor50_az_el[0]:.2f}, el: {sensor50_az_el[1]:.2f}")
    else:
        sensor50_az_el_text.set_val("N/A")

    fig.canvas.draw_idle()

sensors_slider.on_changed(update_sliders)
Q_slider.on_changed(update_sliders)
radius_slider.on_changed(update_sliders)
height_slider.on_changed(update_sliders)

update_sliders(None)
plt.show()
