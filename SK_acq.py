import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.io import loadmat, savemat
import scipy.io as scio
from scipy.fft import fft, ifft, fftshift
from scipy.signal import get_window
import glob
import os
import pandas as pd

def sk_signal(fs, kthr, Data, latest_image_path):
    """
    进行频谱分析和滤波处理。

    参数:
    - fs: 采样频率
    - kthr: 阈值，用于滤波处理
    - Data: 输入数据

    返回:
    - filtered_signal: 滤波后的信号
    - spectrum_image_path: 频谱图的保存路径
    """
    if latest_image_path is None:
        print("latest_image_path 为 None，无法继续执行。")
        return None, None

    folder_path = os.path.dirname(latest_image_path)
    file_basename = os.path.splitext(os.path.basename(latest_image_path))[0] + "_spectrum.jpg"
    spectrum_image_path = os.path.join(folder_path, file_basename)



    # 计算FFT长度（保持为2的幂次方）
    N = len(Data)
    Nfft = 2 ** (np.ceil(np.log2(N)))

    # 创建频率向量
    f = np.linspace(-fs / 2, fs / 2, int(Nfft))
    f1 = f[int(Nfft // 2):]  # 正频率部分

    # 创建时间向量
    t = np.arange(N) / fs

    # FFT分析并居中
    M_az = fftshift(fft(Data, int(Nfft))) / fs

    # FFT computation
    Nfft = 2 ** np.ceil(np.log2(len(Data)))
    f = np.linspace(-fs / 2, fs / 2, int(Nfft))
    f1 = f[int(Nfft // 2):]  # Positive frequencies

    # Time vector
    t_start = 0
    t_end = len(Data) / fs
    t = np.arange(t_start, t_end, 1 / fs)

    # FFT analysis
    M_az = fftshift(fft(Data, int(Nfft))) / fs

    # Plot the FFT magnitude spectrum
    plt.figure()
    plt.plot(f, np.abs(M_az), 'b', linewidth=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f"Spectrum of {latest_image_path}")
    plt.xlim([0, f1[-1]])
    plt.ylim([0, None])
    try:
        # 你的图像生成和保存代码
        plt.savefig(spectrum_image_path)
    except Exception as e:
        print(f"保存图像时出错：{e}")



    # Calculate Spectral Kurtosis (SK)

    wlen = 500  # Window length
    win = get_window('hann', wlen, fftbins=True)
    noverlap = wlen // 2

    # Spectrogram computation, 'f1' is used as frequency array for the spectrogram
    frequencies, times, Sxx = spectrogram(Data, fs=fs, window=win, nperseg=wlen, noverlap=noverlap, nfft=int(Nfft))

    # Reshape Sxx to 2-D array for the following calculations if necessary.
    if len(Sxx.shape) == 1:
        Sxx = Sxx.reshape(-1, 1)

    M_len = Sxx.shape[1]
    Fenzi_M = np.abs(Sxx) ** 4
    Fenmu_M = np.abs(Sxx) ** 2
    K = (M_len / (M_len - 1) * ((M_len + 1) * np.sum(Fenzi_M, axis=0) / (np.sum(Fenmu_M, axis=0) ** 2)) - 2)
    K[K > 0] = 0

    # Updates matching lengths of 'frequencies' and 'K' for plotting
    frequencies_for_K = frequencies[:K.size]

    plt.figure()
    plt.plot(frequencies_for_K, K, 'r', linewidth=2)
    plt.axhline(y=kthr, color='black', linestyle='--', linewidth=2)


    # Filter frequencies below threshold using FFT
    indices_below_thr = np.where(K <= kthr)[0]
    f1_below_thr = frequencies[indices_below_thr]

    S = fft(Data)
    H = np.ones_like(S)

    # Filter processing
    # Implementation may vary depending on the exact requirements
    # ...

    # 应用滤波
    S_filtered = H * S

    # Transform back to time domain
    filtered_signal = np.real(ifft(S_filtered))
    t = t[:len(filtered_signal)]
    # Plot the filtered signal
    plt.figure()
    plt.plot(t, filtered_signal, color='#9300FF', linewidth=1)
    plt.close('all')

    # Save the filtered signal back to .mat file
    # Check for existing variables to be kept in the file and only append 'xdata2'
    # ... 之前的代码 ...

    filtered_signal = np.real(ifft(S_filtered))

    return filtered_signal, spectrum_image_path

