import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.signal import hanning, spectrogram
from scipy.fft import fft, ifft, fftshift
from scipy.signal import get_window
import glob
import os
import pandas as pd
from matplotlib.figure import Figure
from scipy.signal.windows import hann
config = {
    "font.family": 'serif', # 衬线字体
    "font.size": 10, # 相当于小四大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)


def process_and_filter_signal(kthr=-0.715, fs=12000):
    """
    处理并过滤信号的主函数。
    :param kthr: 谱峰度的阈值。
    :param fs: 采样频率。
    """
    folder_path = r'F:\文献'
    processed_files = set()  # 使用集合来存储已处理的文件路径
    figures = []  # 用于存储每个文件的图形

    # 获取当前目录中所有 CSV 文件
    csv_files = glob.glob(folder_path + r'\*.csv')

    # 检查是否找到任何 CSV 文件
    if not csv_files:
        print("未找到任何 CSV 文件。")
    else:
        # 遍历找到的 CSV 文件
        for csv_file_to_read in csv_files:
            # 打印文件路径，用于调试
            print(f"找到文件：{csv_file_to_read}")

            # 如果文件已被处理，跳过它
            if csv_file_to_read in processed_files:
                print(f"文件 {csv_file_to_read} 已经处理过，跳过。")
                continue

            try:
                # 读取选定的 CSV 文件
                df = pd.read_csv(csv_file_to_read, encoding='utf-8')  # 可能需要根据文件实际编码调整
                print(f"已成功读取文件：{csv_file_to_read}")

                # 打印数据框前几行以检查其内容（通常打印前5行）
                print(df.head())

                # 在这里进行文件的处理，例如分析、过滤数据等

                # 处理完成后，将文件路径添加到已处理文件的集合中
                processed_files.add(csv_file_to_read)

            except Exception as e:
                # 如果读取或处理 CSV 文件时出错，打印错误信息
                print(f"处理文件时出错：{csv_file_to_read}")
                print(e)

                # 采样频率和采样时间间隔
                # 采样频率
                fs = 12000
                Ts = 1 / fs

                df = pd.read_csv(csv_file_to_read, encoding='utf-8')
                # 确保 xt 是一个一维 NumPy 数组，代表数据列
                xt = df[df.columns[0]].values
                print(xt)  # 输出第一列振动加速度信号
                # 采样点数
                N = len(xt)
                # 创建图形和轴对象
                plt.plot(np.arange(N) * Ts, xt)
                plt.xlabel('时间（s）')
                plt.ylabel('振动加速度 (m/s^2)')
                plt.title('振动信号时域波形')
                plt.savefig('vibration_signal2.jpg', dpi=300)
                plt.show()
                Data = xt
                fs = 12000  # 采样频率
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
                frequencies, times, Sxx = spectrogram(Data, fs=fs, window=win, nperseg=wlen, noverlap=noverlap,
                                                      nfft=int(Nfft))
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
                plt.show()
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
                plt.show()

                # Save the filtered signal back to .mat file
                # Check for existing variables to be kept in the file and only append 'xdata2'
                # ... 之前的代码 ...

                filtered_signal = np.real(ifft(S_filtered))
                # 获取当前工作目录的路径
                folder_path = os.getcwd()

                # 假设有一个变量 csv_file_to_read 表示你正在处理的CSV文件的全路径
                # 获取CSV文件的基本名称（不包含路径和文件扩展名）
                csv_basename = os.path.basename(csv_file_to_read)
                csv_name_without_ext = os.path.splitext(csv_basename)[0]
                # 假设 filtered_signal 已经是一个一维的 NumPy 数组
                # filtered_signal = ...  # 这是您之前计算得到的滤波后的信号
                # 将 filtered_signal 转换为 DataFrame
                # 如果 filtered_signal 是一维的，我们不需要指定列名，但如果您想要，可以添加一个
                df_filtered_signal = pd.DataFrame(filtered_signal)
                # 构建完整的文件名，包括 _xdata2 后缀和 .csv 扩展名
                full_filename = f"{csv_name_without_ext}_xdata2.csv"
                # 将 DataFrame 保存为 CSV 文件，不包括索引
                df_filtered_signal.to_csv(full_filename, index=False, encoding='utf-8')
                print(f"Filtered signal has been saved to {full_filename}")

