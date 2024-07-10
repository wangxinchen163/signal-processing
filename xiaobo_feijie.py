import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy import signal, fftpack, stats
import pywt
import os
from scipy.io import loadmat
from PyEMD import EMD, Visualisation
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
def xiaobo_feijie_acq(fs, xdata, latest_image_path):
    if latest_image_path is None:
        print("latest_image_path 为 None，无法继续执行。")
        return None, None

    folder_path = os.path.dirname(latest_image_path)
    file_basename = os.path.splitext(os.path.basename(latest_image_path))[0] + "_wavelet.jpg"
    wavelet_image_path = os.path.join(folder_path, file_basename)
    Ts = 1 / fs
    # 采样点数
    N = len(xdata)
    # 去均值，单位方差
    xdata = (xdata - np.mean(xdata)) / np.std(xdata)
    # 设置小波分解的层数
    level = 7
    wavelet = 'dmey'  # 使用dmey小波

    # 进行小波分解
    coeffs = pywt.wavedec(xdata, wavelet, level=level)

    # 从小波分解系数重构信号
    a0 = pywt.waverec(coeffs, wavelet)

    # 绘制原始信号和重构信号及误差
    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.plot(xdata)
    plt.title('Original Signal')

    plt.subplot(3, 1, 2)
    plt.plot(a0)
    plt.title('Reconstructed Signal')

    plt.subplot(3, 1, 3)
    # 重构信号可能会比原始信号长1，若是，去掉最后一个元素
    if len(a0) == len(xdata) + 1:
        a0 = a0[:-1]
    elif len(a0) == len(xdata) - 1:
        # 这情况很少发生，如果发生，则可能需添加一个元素，这里以0填充
        # 但更好的方法是检查重构过程中是否有误
        a0 = np.append(a0, 0)

    # 继续绘制操作
    plt.plot(xdata - a0)

    plt.title('Error Signal')

    # 计算误差
    err = np.max(np.abs(xdata - a0))


    # 从原数据进行小波分解
    coeffs = pywt.wavedec(xdata, 'dmey', level=7)

    # 准备细节信号列表
    details = []  # 存储每一层高频细节信号

    # 遍历每一层来重构细节信号，这里从1开始因为索引0是近似系数
    for i in range(1, len(coeffs)):
        # 创建一个全零系数列表，除了需要的细节系数
        coeff_list = [coeff if index == i else np.zeros_like(coeff) for index, coeff in enumerate(coeffs)]
        # 从修改后的系数列表中重构信号
        detail = pywt.waverec(coeff_list, 'dmey')[:len(xdata)]  # 重构并裁剪到原始数据长度
        # 将重构的细节信号添加到列表
        details.append(detail)

    # 调整顺序使其与MATLAB一致，其中d1是最高频的细节信号
    d7, d6, d5, d4, d3, d2, d1 = details
    # 假定 play_details 包含了所有的细节信号
    play_details = [d1, d2, d3, d4, d5, d6, d7]

    # 只选取前四个细节信号进行绘制
    details_to_plot = play_details[:4]

    # 增大画布尺寸以适应四个子图
    plt.figure(figsize=(4, 6))  # 宽度保持不变，高度适当增加以适应四个子图
    plt.suptitle(f"Wavelet Analysis of {wavelet_image_path}", fontsize=8)  # 调整字体大小为10

    # 为每个选定的细节信号绘制一个子图
    for i, detail in enumerate(details_to_plot, start=1):
        plt.subplot(4, 1, i)  # 注意这里的变化，现在是4个子图，而不是7个
        plt.plot(detail, color=[0/255, 255/255, 0/255])
        plt.title(f'Detail Signal d{i}', fontsize=8)  # 为每个子图设置标题，字号适当调小以适应空间

    # 调整子图间的间距，特别是垂直间距
    plt.subplots_adjust(hspace=0.5)  # 增加垂直间距

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，为总标题留出空间

    try:
        # 你的图像生成和保存代码
        plt.savefig(wavelet_image_path)
    except Exception as e:
        print(f"保存图像时出错：{e}")


    # 第1层高频细节信号的包络谱
    xt = d1
    ht = fftpack.hilbert(xt)
    at1 = np.sqrt(xt ** 2 + ht ** 2)  # 获得解析信号at = sqrt(xt^2 + ht^2)
    am1 = np.fft.fft(at1)  # 对解析信号at做fft变换获得幅值
    am1 = np.abs(am1)  # 对幅值求绝对值（此时的绝对值很大）
    am1 = am1 / len(am1) * 2
    am1[0] = am1[0] / 2
    freq = np.fft.fftfreq(len(at1), d=1 / fs)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(len(freq) / 2)]  # 获取正频率
    plt.close('all')
    return at1, wavelet_image_path