import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.signal import hanning, spectrogram
from scipy.io import loadmat, savemat
import scipy.io as scio
from scipy.fft import fft, ifft, fftshift
from scipy.signal import get_window
from scipy.signal.windows import hann


config = {
    "font.family": 'serif', # 衬线字体
    "font.size": 10, # 相当于小四大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)
def data_acquision(FilePath):
    """
    fun: 从cwru mat文件读取加速度数据
    param file_path: mat文件绝对路径
    return accl_data: 加速度数据，array类型
    """
    data = scio.loadmat(file_path)  # 加载mat数据
    data_key_list = list(data.keys())  # mat文件为字典类型，获取字典所有的键并转换为list类型
    accl_key = data_key_list[3]  # 获取mat中的第四个键（3为DE，4为FE）
    accl_data = data[accl_key].flatten()  # 获取DE/FE振动加速度信号,并将二维数组展成一维数组
    return accl_data
# 采样频率
fs = 12000
Ts = 1 / fs

file_path = r'F:\文献\12k_Drive_End_IR021_0_209.mat'
xt = data_acquision(file_path)
# 采样点数
N = len(xt)
plt.plot(np.arange(N) * Ts, xt)
plt.xlabel('时间（s）')
plt.ylabel('振动加速度 (m/s^2)')
plt.title('振动信号时域波形')
plt.savefig('vibration_signal.png', dpi=300)
plt.show()




Data = xt

fs = 12000  # 采样频率

# 计算FFT长度（保持为2的幂次方）
N = len(Data)
Nfft = 2**(np.ceil(np.log2(N)))

# 创建频率向量
f = np.linspace(-fs/2, fs/2, int(Nfft))
f1 = f[int(Nfft//2):]  # 正频率部分

# 创建时间向量
t = np.arange(N) / fs

# FFT分析并居中
M_az = fftshift(fft(Data, int(Nfft))) / fs

# FFT computation
Nfft = 2**np.ceil(np.log2(len(Data)))
f = np.linspace(-fs/2, fs/2, int(Nfft))
f1 = f[int(Nfft//2):]  # Positive frequencies

# Time vector
t_start = 0
t_end = len(Data) / fs
t = np.arange(t_start, t_end, 1/fs)

# FFT analysis
M_az = fftshift(fft(Data, int(Nfft))) / fs

# Plot the FFT magnitude spectrum
plt.figure()
plt.plot(f, np.abs(M_az), 'b', linewidth=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Spectrum')
plt.xlim([0, f1[-1]])
plt.ylim([0, None])
plt.show()

# Calculate Spectral Kurtosis (SK)
kthr = -0.715  # Threshold
wlen = 500  # Window length
win = get_window('hann', wlen, fftbins=True)
noverlap = wlen // 2

# Spectrogram computation, 'f1' is used as frequency array for the spectrogram
frequencies, times, Sxx = spectrogram(Data, fs=fs, window=win, nperseg=wlen, noverlap=noverlap, nfft=int(Nfft))

# Reshape Sxx to 2-D array for the following calculations if necessary.
if len(Sxx.shape) == 1:
    Sxx = Sxx.reshape(-1, 1)

M_len = Sxx.shape[1]
Fenzi_M = np.abs(Sxx)**4
Fenmu_M = np.abs(Sxx)**2
K = (M_len/(M_len - 1) * ((M_len + 1) * np.sum(Fenzi_M, axis=0) / (np.sum(Fenmu_M, axis=0)**2)) - 2)
K[K > 0] = 0

# Updates matching lengths of 'frequencies' and 'K' for plotting
frequencies_for_K = frequencies[:K.size]

plt.figure()
plt.plot(frequencies_for_K, K, 'r', linewidth=2)
plt.axhline(y=kthr, color='black', linestyle='--', linewidth=2)
plt.show()

# Filter frequencies below threshold using FFT
indices_below_thr = np.where(K <= kthr)[0]
f1_below_thr = frequencies[indices_below_thr]

S = fft(Data)
H = np.ones_like(S)

# Filter processing
# Implementation may vary depending on the exact requirements
# ...

# Apply filter to spectrum
S_filtered = H * S

# Transform back to time domain
filtered_signal = np.real(ifft(S_filtered))
t = t[:len(filtered_signal)]
# Plot the filtered signal
plt.figure()
plt.plot(t, filtered_signal, color='#9300FF', linewidth=1)
plt.show()

# Save the filtered signal back to .mat file
# Check for existing variables to be kept in the file and only append 'xdata2'
# ... 之前的代码 ...

filtered_signal = np.real(ifft(S_filtered))

# 载入原始的.mat文件数据
original_data = scio.loadmat(file_path)

# 将过滤后的信号添加到原始数据字典中，使用新键 'xdata2'
original_data['xdata2'] = filtered_signal

# 保存更新后的数据到.mat文件，现在包括所有原始变量和新变量 'xdata2'
savemat(file_path, original_data)

