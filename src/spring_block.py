import numpy as np  # 导入numpy库，用于数学计算
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘制图形
from scipy.integrate import odeint  # 导入odeint函数，用于求解微分方程


def solve_ode_euler(step_num):
    """
    使用欧拉法求解弹簧 - 质点系统的常微分方程。

    参数:
    step_num (int): 模拟的步数

    返回:
    tuple: 包含时间数组、位置数组和速度数组的元组
    """
    # 创建存储位置和速度的数组，长度为 step_num + 1
    position = np.zeros(step_num + 1)  # 创建一个数组来存储位置
    velocity = np.zeros(step_num + 1)  # 创建一个数组来存储速度

    # 计算时间步长
    time_step = 2 * np.pi / step_num  # 计算每一步的时间长度

    # 设置初始位置和速度
    position[0] = 0  # 初始位置设为0
    velocity[0] = 1  # 初始速度设为1

    # 使用欧拉法迭代求解微分方程
    for i in range(step_num):
        # 根据微分方程更新位置
        position[i + 1] = position[i] + velocity[i] * time_step  # 用速度和时间步长更新位置
        # 根据微分方程更新速度，这里假设 k = m = 1
        velocity[i + 1] = velocity[i] - position[i] * time_step  # 用位置和时间步长更新速度

    # 生成时间数组
    time_points = np.arange(step_num + 1) * time_step  # 创建一个时间数组

    return time_points, position, velocity  # 返回时间、位置和速度


def spring_mass_ode_func(state, time):
    """
    定义弹簧 - 质点系统的常微分方程。

    参数:
    state (list): 包含位置和速度的列表
    time (float): 时间

    返回:
    list: 包含位置和速度的导数的列表
    """
    position, velocity = state  # 从状态中提取位置和速度
    d_position_dt = velocity  # 位置的导数是速度
    d_velocity_dt = -position  # 速度的导数是负的位置（假设k = m = 1）
    return [d_position_dt, d_velocity_dt]  # 返回位置和速度的导数


def solve_ode_odeint(step_num):
    """
    使用 odeint 求解弹簧 - 质点系统的常微分方程。

    参数:
    step_num (int): 模拟的步数

    返回:
    tuple: 包含时间数组、位置数组和速度数组的元组
    """
    # 初始条件
    initial_state = [0, 1]  # 初始位置和速度
    # 时间点
    time_points = np.linspace(0, 2 * np.pi, step_num + 1)  # 创建时间点数组
    # 使用 odeint 求解微分方程
    solution = odeint(spring_mass_ode_func, initial_state, time_points)  # 调用odeint求解微分方程
    position = solution[:, 0]  # 提取位置
    velocity = solution[:, 1]  # 提取速度
    return time_points, position, velocity  # 返回时间、位置和速度


def plot_ode_solutions(time_euler, position_euler, velocity_euler, time_odeint, position_odeint, velocity_odeint):
    """
    绘制欧拉法和 odeint 求解的位置和速度随时间变化的图像。

    参数:
    time_euler (np.ndarray): 欧拉法的时间数组
    position_euler (np.ndarray): 欧拉法的位置数组
    velocity_euler (np.ndarray): 欧拉法的速度数组
    time_odeint (np.ndarray): odeint 的时间数组
    position_odeint (np.ndarray): odeint 的位置数组
    velocity_odeint (np.ndarray): odeint 的速度数组
    """
    plt.figure(figsize=(12, 6))  # 创建一个大小为12x6的图形

    # 绘制位置对比图
    plt.subplot(1, 2, 1)  # 创建第一个子图
    plt.plot(time_euler, position_euler, 'ro', label='Euler Position')  # 用红色圆点绘制欧拉法的位置
    plt.plot(time_odeint, position_odeint, 'b-', label='ODEint Position')  # 用蓝色实线绘制odeint的位置
    plt.xlabel('Time')  # 设置x轴标签为时间
    plt.ylabel('Position')  # 设置y轴标签为位置
    plt.title('Position Comparison')  # 设置标题为位置对比
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格

    # 绘制速度对比图
    plt.subplot(1, 2, 2)  # 创建第二个子图
    plt.plot(time_euler, velocity_euler, 'gs', label='Euler Velocity')  # 用绿色方块绘制欧拉法的速度
    plt.plot(time_odeint, velocity_odeint, 'm-', label='ODEint Velocity')  # 用紫色实线绘制odeint的速度
    plt.xlabel('Time')  # 设置x轴标签为时间
    plt.ylabel('Velocity')  # 设置y轴标签为速度
    plt.title('Velocity Comparison')  # 设置标题为速度对比
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格

    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图形


if __name__ == "__main__":
    # 模拟步数
    step_count = 100  # 设置模拟步数为100
    # 欧拉法求解
    time_euler, position_euler, velocity_euler = solve_ode_euler(step_count)  # 调用欧拉法求解
    # odeint 求解
    time_odeint, position_odeint, velocity_odeint = solve_ode_odeint(step_count)  # 调用odeint求解
    # 绘制对比结果
    plot_ode_solutions(time_euler, position_euler, velocity_euler, time_odeint, position_odeint, velocity_odeint)  # 绘制结果对比图
