import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
def sineWaveZeroPhi(x, t, A, omega, k):
    sin=np.sin(k*x-omega*t)
    return A*sin
# 创建动画所需的 Figure 和 Axes
fig = plt.figure(figsize=(10,6))
subplot = plt.axes(xlim=(0, 10), xlabel="x", ylim=(-2, 2), ylabel="y")

# 创建空的line对象，用于动画显示
line1, = subplot.plot([], [], lw=2)
line2, = subplot.plot([], [], lw=2)
line3, = subplot.plot([], [], lw=2)
lines = [line1, line2, line3]
# 创建一个line对象列表，便于操作

def init():
    for line in lines:
        line.set_data([],[])
    return lines
x=np.linspace(0,10,1000)
def animate(i):
    A=1
    omega=2*np.pi
    k=np.pi/2
    t=0.01*i
    y1=sineWaveZeroPhi(x,t,A, omega, k)
    y2=sineWaveZeroPhi(x,t, A, omega, -k)
    y3=y1+y2
    lines[0].set_data(x,y1)
    lines[1].set_data(x, y2)
    lines[2].set_data(x, y3)
    return lines
if __name__ == '__main__':
    ani=animation.FuncAnimation(fig,animate,init_func=init,frames=200,interval=20,blit=True)
    plt.tight_layout()
    plt.show()

