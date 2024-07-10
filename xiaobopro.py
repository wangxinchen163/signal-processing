import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy import signal, fftpack, stats
import pywt
from scipy.io import loadmat
from PyEMD import EMD, Visualisation
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


config = {
    "font.family": 'serif', # 衬线字体
    "font.size": 10, # 相当于小四大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)
# 加载MATLAB .mat文件


mat_contents = loadmat(r'F:\文献\12k_Drive_End_IR021_0_209.mat')
xdata = mat_contents['xdata2'].flatten()

# 采样频率
fs = 12000
Ts = 1 / fs

# 采样点数
N = len(xdata)

# 去均值，单位方差
xdata = (xdata - np.mean(xdata)) / np.std(xdata)

# 信号时域波形
plt.figure(1)
plt.plot(np.arange(N) * Ts, xdata)
plt.xlabel('时间（s）')
plt.ylabel('振动加速度 (m/s^2)')
plt.title('振动信号时域波形')
plt.savefig('vibration_signal.png', dpi=300)
plt.show()
# 假设 xdata 已经是您的原始信号，并且 N 是信号的长度
N = len(xdata)
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

plt.show()
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

# 增大画布尺寸
plt.figure(figsize=(4, 8))  # 宽度保持不变，高度增加

# 为每个细节信号绘制一个子图
for i, detail in enumerate(play_details, start=1):
    plt.subplot(7, 1, i)
    plt.plot(detail)
    plt.title(f'd{i}')

plt.tight_layout()  # 自动调整子图参数，优化布局
plt.show()
def bearing_fault_freq_cal(n, d, D, alpha, fr=None):
    '''
    基本描述：
        计算滚动轴承的故障特征频率
    详细描述：
        输入4个参数 n, fr, d, D, alpha
    return C_bpfi, C_bpfo, C_bsf, C_ftf,  fr
           内圈    外圈    滚针   保持架  转速

    Parameters
    ----------
    n: integer
        The number of roller element
    fr: float(r/min)
        Rotational speed
    d: float(mm)
        roller element diameter
    D: float(mm)
        pitch diameter of bearing
    alpha: float(°)
        contact angle
    fr:：float(r/min)
        rotational speed
    Returns
    -------
    BPFI: float(Hz)
        Inner race-way fault frequency
    BPFO: float(Hz)
        Outer race-way fault frequency
    BSF: float(Hz)
        Ball fault frequency
    FTF: float(Hz)
        Cage frequency
    '''
    C_bpfi = n*(1/2)*(1+d/D*np.math.cos(alpha))
    C_bpfo = n*(1/2)*(1-(d/D)*np.math.cos(alpha))
    C_bsf = D*(1/(1*d))*(1-np.square(d/D*np.math.cos(alpha)))
    C_ftf = (1/2)*(1-(d/D)*np.math.cos(alpha))
    if fr!=None:
        return C_bpfi*fr/60, C_bpfo*fr/60, C_bsf*fr/60, C_ftf*fr/60, fr/60
    else:
        return C_bpfi, C_bpfo, C_bsf, C_ftf, fr
bpfi, bpfo, bsf, ftf, fr = bearing_fault_freq_cal(n=9, alpha=0, d=7.94, D=39.04, fr=1797)
print('内圈故障特征频率',bpfi)
print('外圈故障特征频率',bpfo)
print('滚动体故障特征频率',bsf)
print('保持架故障特征频率',ftf)
print('转动频率',fr)
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
# 计算索引范围对应于 0-300 Hz
idx = np.where((freq >= 0) & (freq <= 300))[0]

# 选取0-300hz的频率和对应的振幅
selected_freq = freq[idx]
selected_am1 = am1[idx]

# 绘图
plt.plot(selected_freq, selected_am1)
# 假设您的信号存储在变量 'at' 中
# 在这里 'at' 应该是一个 numpy 数组，而不是直接赋值的 'at'
x = at1  # 把 'at' 替换成您自己的信号

# 初始化和配置EMD分析器
emd = EMD(spline_kind='cubic', extrema_detection="simple")

# 执行经验模态分解
imfs = emd(x)

# 'imfs' 是分解得到的内禀模态函数（Intrinsic Mode Functions）
# 'residual' 是残差，它等于原始信号减去所有的 IMFs
residual = x - np.sum(imfs, axis=0)
vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=residual, t=np.arange(len(x)), include_residue=True)
vis.show()
imf1 = imfs[0, :]
xt = imf1  # 取出第一个 IMF
xt = xt - np.mean(xt)  # 从 IMF 减去其平均值
# ----做希尔伯特变换----#
ht = fftpack.hilbert(xt)
at = np.sqrt(xt ** 2 + ht ** 2)  # 获得解析信号at = sqrt(xt^2 + ht^2)
am = np.fft.fft(at)  # 对解析信号at做fft变换获得幅值
am = np.abs(am)  # 对幅值求绝对值（此时的绝对值很大）
am = am / len(am) * 2
am = am[0: int(len(am) / 2)]  # 取正频率幅值
# 找到频率小于 1 Hz 的所有索引
indices = np.where(freq < 2)
# 将频率小于 1 Hz 对应的 am 值置为 0
am[indices] = 0

freq = np.fft.fftfreq(len(at), d=1 / fs)  # 获取fft频率，此时包括正频率和负频率
freq = freq[0:int(len(freq) / 2)]  # 获取正频率
plt.plot(freq, am)
plt.show()
# 使用布尔索引来选择频率在0-300Hz范围内的数据
mask = (freq >= 0) & (freq <= 300)

# 只选择0-300 Hz的振幅值和频率值
freq_filtered = freq[mask]
am_filtered = am[mask]

# 绘制筛选后的数据
plt.plot(freq_filtered, am_filtered)

# 设置 x 轴和 y 轴的标签
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('故障诊断结果图')
# 获取当前的y轴范围
current_ylim = plt.ylim()

# 在图表中添加垂直虚线
h1 = plt.axvline(x=bpfi, color='g', linewidth=1.5, linestyle='--', label='BPFI')
h2 = plt.axvline(x=bpfo, color='k', linewidth=1.5, linestyle='--', label='BPFO')
h3 = plt.axvline(x=bsf, color='r', linewidth=1.5, linestyle='--', label='BSF')

# 设置字体大小
plt.gca().tick_params(axis='both', which='major', labelsize=13)

# 添加图例
plt.legend()
# 保存图表为PNG文件，设置分辨率
plt.savefig('fault_diagnosis.png', dpi=300)
# 显示图表
plt.show()
plt.show()

# 假设您已经拥有了以下变量
# freq: 所有频率值的数组
# am: 对应的振幅值数组
# 确定特征频率以及频率窗口
characteristic_freqs = {'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf}
frequency_window = 10  # 特征频率周围的搜索窗口宽度，例如：±10 Hz

# 创建一个掩码来筛选想要的频率范围和振幅
mask = (freq >= 60) & (freq <= 300)

# 只选择0-300 Hz的振幅值和频率值
freq_filtered = freq[mask]
am_filtered = am[mask]

# 绘制筛选后的数据
plt.plot(freq_filtered, am_filtered)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Filtered Frequency Domain Data (50Hz-300Hz)')

# 寻找峰值
peaks, _ = find_peaks(am_filtered)

# 测量峰值的小窗口宽度±3Hz和大窗口宽度±20Hz
peak_frequency_window = 2
average_frequency_window = 20

# 特征故障频率字典
characteristic_freqs = {'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf}

# 存储所有检测到的故障信息
total_faults = []
total_fault_detected = False

# 迭代每一种特征故障频率
for label, f_target in characteristic_freqs.items():
    # 决定特征频率周围的大窗口频率范围来计算平均振幅
    average_window_freqs = freq_filtered[(freq_filtered >= f_target - average_frequency_window) &
                                         (freq_filtered <= f_target + average_frequency_window)]
    average_window_peak_indices = [i for i, f in enumerate(freq_filtered) if f in average_window_freqs]
    average_window_amplitudes = am_filtered[average_window_peak_indices]

    # 计算大窗口内的平均振幅
    window_mean_amplitude = np.mean(average_window_amplitudes) if average_window_amplitudes.size > 0 else 0

    # 找出特征频率±3Hz内的峰值
    peak_indices = np.where((freq_filtered[peaks] >= f_target - peak_frequency_window) &
                            (freq_filtered[peaks] <= f_target + peak_frequency_window))[0]

    faults = []
    fault_detected = False

    # 检测每个峰值是否代表故障
    for peak_index in peak_indices:
        peak_freq = freq_filtered[peaks[peak_index]]
        peak_amplitude = am_filtered[peaks[peak_index]]
        if peak_amplitude > window_mean_amplitude * 8:
            fault_info = f"故障特征频率为 {peak_freq} Hz（故障类型为{label}）"
            faults.append(fault_info)
            fault_detected = True

    # 如果检测到故障，则将故障信息添加到总列表中
    if fault_detected:
        total_fault_detected = True
        total_faults.extend(faults)

# 所有故障类型检测完成后，输出最终结论
print("全部故障检测完成，结论一:", ("有故障" if total_fault_detected else "无故障"))
print("全部故障信息：", "；".join(total_faults) if total_fault_detected else "无故障信息")

# 确定最终结论以用在报告中
final_conclusion_1 = "有故障" if total_fault_detected else "无故障"
final_conclusion_2 = "；".join(total_faults) if total_fault_detected else "无故障"

# 展示振动信号和故障诊断结果图
plt.figure()
plt.title("Vibration Signal and Fault Diagnosis Result")
# 假定您有绘图代码
# plt.plot(...)
plt.show()

# 生成HTML报告
html_content = f'''
<html>
<head>
    <meta charset="UTF-8">
    <title>报告</title>
</head>
<body>
    <h1>基于改进小波分解的海上风机轴承故障诊断结果报告</h1>
    <p>时域波形图和故障诊断结果图</p>

    <img src="vibration_signal.png" alt="振动信号时域波形" style="max-width: 50%; height: auto;">
    <img src="fault_diagnosis.png" alt="故障诊断结果图" style="max-width: 50%; height: auto;">

    <p>结论一: {final_conclusion_1}</p>
    <p>结论二: {final_conclusion_2}</p>
</body>
</html>
'''

with open('report.html', 'w', encoding='utf-8') as file:
    file.write(html_content)
print("HTML报告已经生成。")
