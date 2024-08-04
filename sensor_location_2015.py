import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

# 初始化参数
target_position = np.array([1000, 1000, 1000])  # 静态目标位置
sigma_u = 0.175  # 测量噪声的单位距离误差 1度是0.0175弧度，是这样吗？ 但是我用这个值才能跑出论文的样子
gamma = 0.2  # 功率衰减指数

# 状态转移矩阵 F_k
T = 2  # 时间间隔
F = np.array([
    [1, T, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, T, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, T],
    [0, 0, 0, 0, 0, 1]
])

# 这里是：将预测位置与sensor位置的关系转化成方位角与仰角
# EKF 预测和更新步骤
def h(X, sensor_position):
    delta_x = X[0] - sensor_position[0]
    delta_y = X[2] - sensor_position[1]
    delta_z = X[4] - sensor_position[2]
    azimuth = np.arctan2(delta_y, delta_x)
    elevation = np.arctan2(delta_z, np.sqrt(delta_x ** 2 + delta_y ** 2))
    return np.array([azimuth, elevation])

# 这里是h的雅可比矩阵，K和P的运算会用到
def H_jacobian(X, sensor_position):
    delta_x = X[0] - sensor_position[0]
    delta_y = X[2] - sensor_position[1]
    delta_z = X[4] - sensor_position[2]
    d_xy = delta_x ** 2 + delta_y ** 2
    d_xyz = d_xy + delta_z ** 2
    sqrt_d_xy = np.sqrt(d_xy)

    H = np.zeros((2, 6))
    H[0, 0] = -delta_y / d_xy
    H[0, 2] = delta_x / d_xy
    H[1, 0] = -delta_x * delta_z / (d_xyz * sqrt_d_xy)
    H[1, 2] = -delta_y * delta_z / (d_xyz * sqrt_d_xy)
    H[1, 4] = sqrt_d_xy / d_xyz

    return H

# 论文里没说Q的事..设置成0？
def predict(X, P, F, Q):
    X = F @ X
    P = F @ P @ F.T + Q
    return X, P

def update(X, P, z_noisy, H, R, sensor_position):
    y = z_noisy - h(X, sensor_position)     # z_n是带噪声的测量值
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    X = X + K @ y
    P = (np.eye(6) - K @ H) @ P
    return X, P

# 模拟传感器测量角度并加噪声
def measurement(X, sensor_position, R):
    M = np.array([X[0], 0, X[1], 0, X[2], 0])   # target的位置的结构和h用的X不一样，所以在这转变一下
    true_measurement = h(M, sensor_position)
    noise = np.random.multivariate_normal([0, 0], R)
    noisy_measurement = true_measurement + noise    # 加了噪音，所以不同传感器测出的仰角不同，正常。
    return noisy_measurement


def simulate_ekf(N, Q_val, iterations, radius, height):
    # 初始化传感器位置
    sensor_positions = np.zeros((N, 3))  # N 是行，传感器个数
    for k in range(N):  # 分配每个传感器的位置, 由于是仿真，直接由target真实位置得到sensor的位置，确保target在中轴线上
        theta_k = 2 * np.pi * k / N
        sensor_positions[k, 0] = target_position[0] + radius * np.cos(theta_k)  # sensor's x
        sensor_positions[k, 1] = target_position[1] + radius * np.sin(theta_k)  # sensor's y
        sensor_positions[k, 2] = height  # sensor's z

    # 初始化状态向量和协方差矩阵
    X = np.array([1200, 0, 800, 0, 1400, 0])  # 初始状态
    P = np.diag([100 ** 2, 0, 100 ** 2, 0, 100 ** 2, 0])  # 初始协方差矩阵有较小的不确定性
    Q = np.eye(6) * Q_val
    mse_list = []

    sensor1_az_el = None
    sensor50_az_el = None

    for i in range(iterations):     # 遍历每个传感器
        for k in range(N):
            # 预测步骤
            X, P = predict(X, P, F, Q)

            # 传感器位置
            sensor_position = sensor_positions[k]

            # 计算传感器到目标的距离
            d_k = np.linalg.norm(X[[0, 2, 4]] - sensor_position)

            # 根据公式计算测量噪声协方差矩阵 R_k
            R_k = np.diag([sigma_u ** 2 * d_k ** gamma, sigma_u ** 2 * d_k ** gamma])

            # 测量步骤
            z_noisy = measurement(np.concatenate([target_position, np.zeros(3)]), sensor_position, R_k)

            # 计算雅可比矩阵
            H = H_jacobian(X, sensor_position)

            # 更新步骤
            X, P = update(X, P, z_noisy, H, R_k, sensor_position)

            # 保存传感器1和传感器50的方位角和仰角
            if k == 0:
                sensor1_az_el = np.degrees(z_noisy)  # 转换为角度
                sensor1_position = sensor_position
            elif k == 49 and N >= 50:
                sensor50_az_el = np.degrees(z_noisy)  # 转换为角度
                sensor50_position = sensor_position

        mse = np.trace(P)
        mse_list.append(mse)

    print(f"MSE after {iterations} iterations: {mse_list[-1]}")
    return sensor_positions, mse_list, sensor1_az_el, sensor50_az_el, sensor1_position, sensor50_position

# 创建图形和滑块
fig = plt.figure(figsize=(15, 10))
ax_3d = fig.add_subplot(121, projection='3d')
ax_mse = fig.add_subplot(122)
plt.subplots_adjust(left=0.25, bottom=0.35)

ax_sensors = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_Q = plt.axes([0.1, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_radius = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_height = plt.axes([0.1, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')

sensors_slider = Slider(ax_sensors, 'Sensors', 10, 200, valinit=100, valstep=1)
Q_slider = Slider(ax_Q, 'Process Noise (Q)', 0, 1.0, valinit=0, valstep=0.1)  # 论文没说Q的事，先设成0
radius_slider = Slider(ax_radius, 'Radius', 4, 2000, valinit=591.80, valstep=1)
height_slider = Slider(ax_height, 'Height', 0, 2000, valinit=462.38, valstep=1)

# 调整显示传感器方位角和仰角的位置和大小
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

    sensor_positions, mse_list, sensor1_az_el, sensor50_az_el, sensor1_position, sensor50_position = simulate_ekf(N, Q_val, 100, radius, height)

    sensor_history.append(sensor_positions)
    mse_history.append(mse_list)

    ax_3d.clear()
    ax_mse.clear()

    # 显示所有历史传感器位置
    colors = plt.cm.viridis(np.linspace(0, 1, len(sensor_history)))
    for i, sensors in enumerate(sensor_history):
        ax_3d.scatter(sensors[:, 0], sensors[:, 1], sensors[:, 2], color=colors[i], label=f'State {i + 1}', alpha=0.6)

    # 用五角星表示 target 和两个 sensor
    ax_3d.scatter(target_position[0], target_position[1], target_position[2], marker='*', label='True Target Position', color='green', s=200)
    ax_3d.scatter(sensor1_position[0], sensor1_position[1], sensor1_position[2], marker='*', label='Sensor 1', color='red', s=200)
    ax_3d.scatter(sensor50_position[0], sensor50_position[1], sensor50_position[2], marker='*', label='Sensor 50', color='blue', s=200)

    # 显示坐标
    ax_3d.text(sensor1_position[0], sensor1_position[1], sensor1_position[2], f"{sensor1_position[0]:.1f}, {sensor1_position[1]:.1f}, {sensor1_position[2]:.1f}", color='red')
    ax_3d.text(sensor50_position[0], sensor50_position[1], sensor50_position[2], f"{sensor50_position[0]:.1f}, {sensor50_position[1]:.1f}, {sensor50_position[2]:.1f}", color='blue')

    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend()

    for i, mse_list in enumerate(mse_history):
        if any(mse_list):  # 仅绘制非零的 MSE
            ax_mse.plot(range(1, 101), mse_list, color=colors[i], label=f'State {i + 1} MSE')

    # 动态设置 y 轴范围并应用对数刻度
    all_mse = np.concatenate(mse_history)
    if len(all_mse) > 0:
        min_mse = all_mse.min()
        max_mse = all_mse.max()
        # 避免在对数刻度下出现0
        min_mse = max(min_mse, 1e-10)  # 设定最小值
        ax_mse.set_ylim([min_mse * 0.9, max_mse * 1.1])  # 动态调整 y 轴范围
        ax_mse.set_yscale('log')  # 设置 y 轴为对数刻度

    ax_mse.set_xlabel('Iteration')
    ax_mse.set_ylabel('MSE')
    ax_mse.legend()

    sensor1_az_el_text.set_val(f"az: {sensor1_az_el[0]:.2f}, el: {sensor1_az_el[1]:.2f}")
    if sensor50_az_el is not None:
        sensor50_az_el_text.set_val(f"az: {sensor50_az_el[0]:.2f}, el: {sensor50_az_el[1]:.2f}")
    else:
        sensor50_az_el_text.set_val("N/A")

    # 显示 MSE 的起始值和最终值
    ax_mse.text(0, mse_list[0], f"{mse_list[0]:.2f}", color='red', fontsize=12)
    ax_mse.text(99, mse_list[-1], f"{mse_list[-1]:.2f}", color='red', fontsize=12)

    fig.canvas.draw_idle()

sensors_slider.on_changed(update_sliders)
Q_slider.on_changed(update_sliders)
radius_slider.on_changed(update_sliders)
height_slider.on_changed(update_sliders)

# 初始化显示
update_sliders(None)
plt.show()
