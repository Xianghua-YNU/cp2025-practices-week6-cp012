# 导入必要的库
import numpy as np  # 数值计算库，用于处理数组和数学运算
import matplotlib.pyplot as plt  # 绘图库，用于数据可视化


# =================== 任务1：拍频现象的数值模拟 ===================
def simulate_beat_frequency(
        f1=440,  # 第一个波的频率，默认440Hz（标准音A4）
        f2=444,  # 第二个波的频率，默认444Hz
        A1=1.0,  # 第一个波的振幅，默认1.0
        A2=1.0,  # 第二个波的振幅，默认1.0
        t_start=0,  # 时间起点，默认0秒
        t_end=1,  # 时间终点，默认1秒
        num_points=5000,  # 采样点数，默认5000个点
        show_plot=True  # 是否显示图表，默认显示
):
    """模拟两个正弦波叠加产生拍频现象的函数"""

    # 生成时间数组：从起始时间到结束时间，生成等间距的时间点
    t = np.linspace(t_start, t_end, num_points)  # 生成5000个时间点的一维数组

    # 用数学公式生成两个正弦波形（使用NumPy的矢量化运算）
    wave1 = A1 * np.sin(2 * np.pi * f1 * t)  # 第一个波：振幅×正弦函数(2π频率×t)
    wave2 = A2 * np.sin(2 * np.pi * f2 * t)  # 第二个波，同上
    superposed_wave = wave1 + wave2  # 叠加两个波形（数组对应位置相加）

    # 计算拍频：两个频率的绝对差值
    beat_frequency = abs(f1 - f2)  # 这个结果会是肉眼可见的包络波动频率

    # 绘制图表部分（当show_plot参数为True时执行）
    if show_plot:
        # 创建一个12英寸宽，6英寸高的画布
        plt.figure(figsize=(12, 6))

        # 第一个子图：显示原始波1 --------------------------------------------------
        plt.subplot(3, 1, 1)  # 创建3行1列的子图矩阵，当前选择第1个位置
        plt.plot(t, wave1, label=f'Wave 1: {f1} Hz')  # 画第一个波形曲线
        plt.title('Original Wave 1')  # 设置标题
        plt.xlabel('Time (s)')  # X轴标签（时间）
        plt.ylabel('Amplitude')  # Y轴标签（振幅）
        plt.legend()  # 显示图注（在label参数中定义的内容）
        plt.grid(True)  # 显示网格线

        # 第二个子图：显示原始波2 --------------------------------------------------
        plt.subplot(3, 1, 2)  # 选择3行1列中的第2个位置
        plt.plot(t, wave2, label=f'Wave 2: {f2} Hz', color='orange')
        plt.title('Original Wave 2')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        # 第三个子图：显示叠加波形 --------------------------------------------------
        plt.subplot(3, 1, 3)  # 选择3行1列中的第3个位置
        plt.plot(t, superposed_wave, label='Superposed Wave', color='green')
        plt.title('Superposed Wave with Beat Frequency')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        # 在图中添加文字标注拍频数值
        plt.text(
            0.5, 0.5,  # 文字位置的归一化坐标（0.5,0.5表示画面中心）
            f'Beat Frequency: {beat_frequency} Hz',
            transform=plt.gca().transAxes,  # 使用坐标轴系作为基准
            ha='center', va='center',  # 水平居中、垂直居中
            bbox=dict(facecolor='white', alpha=0.8)  # 添加半透明白色底框
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()  # 自动调整子图间距防止重叠
        plt.show()  # 显示全部图表

    # 返回生成的时间序列、叠加波形和计算得到的拍频值
    return t, superposed_wave, beat_frequency


# =================== 任务2：参数敏感性分析 ===================
def parameter_sensitivity_analysis():
    """分析不同参数对拍频现象影响的函数"""

    # 第一部分：分析不同频率差的影响 -------------------------------
    plt.figure(1, figsize=(12, 8))  # 创建编号为1的图表，设置画布大小
    base_freq = 440  # 基准频率440Hz
    delta_freqs = [1, 2, 5, 10, 20]  # 频率差列表（1Hz、2Hz等）
    t_end = 0.5  # 缩短显示的时长（0.5秒）以便更清晰地观察拍频

    # 绘制不同频率差对应的叠加波形子图
    for i, delta in enumerate(delta_freqs):  # 遍历频率差列表（i是索引，delta是当前值）
        f2 = base_freq + delta  # 计算测试的第二个频率
        # 调用模拟函数（show_plot=False 表示不单独显示每个波形）
        t, superposed_wave, beat_freq = simulate_beat_frequency(
            f1=base_freq, f2=f2, t_end=t_end, show_plot=False)
        # 创建子图位置（3行2列的布局，i从0开始所以+1调整位置）
        plt.subplot(3, 2, i + 1)
        plt.plot(t, superposed_wave)
        plt.title(f'Beat Frequency: {beat_freq} Hz')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

    plt.tight_layout()  # 调整子图间距
    plt.suptitle('Effect of Frequency Difference on Beat Phenomenon')  # 总标题
    plt.subplots_adjust(top=0.9)  # 调整顶部留空位置确保标题可见
    plt.show()

    # 第二部分：分析不同振幅比例的影响 -------------------------------
    plt.figure(2, figsize=(12, 8))  # 创建编号为2的新图表
    f1, f2 = 440, 444  # 固定两个频率
    amplitude_ratios = [0.1, 0.5, 1.0, 2.0, 5.0]  # 振幅比例列表（波2/波1的比率）

    # 绘制不同振幅比例如应的波形子图
    for i, ratio in enumerate(amplitude_ratios):
        A1, A2 = 1.0, ratio  # A1保持1.0，A2按比例变化
        t, superposed_wave, _ = simulate_beat_frequency(
            f1=f1, f2=f2, A1=A1, A2=A2, t_end=t_end, show_plot=False)
        plt.subplot(3, 2, i + 1)
        plt.plot(t, superposed_wave)
        plt.title(f'Amplitude Ratio: {A1}:{A2}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle('Effect of Amplitude Ratio on Beat Phenomenon')  # 总标题
    plt.subplots_adjust(top=0.9)
    plt.show()


# =================== 主程序入口 ===================
if __name__ == "__main__":  # 当直接运行本脚本时执行以下代码
    print("=== 任务1: 基本拍频模拟 ===")
    # 默认参数调用（会显示完整的三幅子图）
    t, wave, beat_freq = simulate_beat_frequency()
    print(f"计算得到的拍频为: {beat_freq} Hz\n")  # \n是换行符

    print("=== 任务2: 参数敏感性分析 ===")
    # 执行参数分析（会显示两组参数变化的对比图）
    parameter_sensitivity_analysis()
