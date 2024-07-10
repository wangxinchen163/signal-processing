import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy import signal, fftpack, stats
import pywt
from zdyPyEMD import zdyEMD
from scipy.io import loadmat
from PyEMD import EMD, Visualisation
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
def extract_freq_from_label(label_text):
    """从标签文本中提取频率值"""
    # 假设标签文本格式为 '123.45 Hz'
    freq_str = label_text.split()[0]  # 分割字符串并获取第一部分
    try:
        bfreq = float(freq_str)  # 尝试将字符串转换为浮点数
    except ValueError:
        bfreq = None  # 如果转换失败，返回 None
    return bfreq
def emd_feijie_acq(fs, at1, latest_image_path, result_labels):
    if latest_image_path is None:
        print("latest_image_path 为 None，无法继续执行。")
        return None, None
    # 从result_labels中提取频率
    bpfi = extract_freq_from_label(result_labels["bpfi"].cget("text"))
    bpfo = extract_freq_from_label(result_labels["bpfo"].cget("text"))
    bsf = extract_freq_from_label(result_labels["bsf"].cget("text"))

    folder_path = os.path.dirname(latest_image_path)
    file_basename = os.path.splitext(os.path.basename(latest_image_path))[0] + "_EMD.jpg"
    file_basename2 = os.path.splitext(os.path.basename(latest_image_path))[0] + "_fault_diagnosis.jpg"
    emd_image_path = os.path.join(folder_path, file_basename)
    fault_diagnosis_image_path = os.path.join(folder_path, file_basename2)
    x = at1  # 把 'at' 替换成您自己的信号

    # 初始化和配置EMD分析器
    emd = EMD(spline_kind='cubic', extrema_detection="simple")

    # 执行经验模态分解
    imfs = emd(x)
    imf1 = imfs[0, :]
    imf2 = imfs[1, :]
    imf3 = imfs[2, :]
    imf4 = imfs[3, :]
    imf5 = imfs[4, :]
    imf6 = imfs[5, :]
    imf7 = imfs[6, :]
    n = 7
    plt.figure(figsize=(10, 2 * n))  # 调整画布大小以适应n个子图，每个IMF高度设置为2
    plt.suptitle(f"EMD of {emd_image_path}", fontsize=10)  # 设置总标题和字体大小

    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(imfs[i, :], label=f'IMF{i + 1}', color=[255/255, 69/255, 0/255])
        plt.title(f'IMF{i + 1}', fontsize=10)
        plt.legend()
    plt.subplots_adjust(hspace=0.5)  # 增加垂直间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，为总标题留出空间
    try:
        # 你的图像生成和保存代码
        plt.savefig(emd_image_path)
        plt.close()
    except Exception as e:
        print(f"保存图像时出错：{e}")


    xt = imf1  # 取出第一个 IMF
    xt = xt - np.mean(xt)  # 从 IMF 减去其平均值
    # ----做希尔伯特变换----#
    ht = fftpack.hilbert(xt)
    at = np.sqrt(xt ** 2 + ht ** 2)  # 获得解析信号at = sqrt(xt^2 + ht^2)
    am = np.fft.fft(at)  # 对解析信号at做fft变换获得幅值
    am = np.abs(am)  # 对幅值求绝对值（此时的绝对值很大）
    am = am / len(am) * 2
    am = am[0: int(len(am) / 2)]  # 取正频率幅值


    freq = np.fft.fftfreq(len(at), d=1 / fs)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(len(freq) / 2)]  # 获取正频率
    # 找到频率小于 1 Hz 的所有索引
    indices = np.where(freq < 2)
    # 将频率小于 1 Hz 对应的 am 值置为 0
    am[indices] = 0

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
    plt.title(f"fault_diagnosis of {fault_diagnosis_image_path}")
    # 获取当前的y轴范围
    current_ylim = plt.ylim()
    # 确保所有频率都已经被正确提取
    if bpfi is not None and bpfo is not None and bsf is not None:
        # 在图表中添加垂直虚线
        h1 = plt.axvline(x=bpfi, color='g', linewidth=1.5, linestyle='--', label='BPFI')
        h2 = plt.axvline(x=bpfo, color='k', linewidth=1.5, linestyle='--', label='BPFO')
        h3 = plt.axvline(x=bsf, color='r', linewidth=1.5, linestyle='--', label='BSF')
    else:
        print("一个或多个特征频率值为空，跳过添加垂直虚线的操作。")
    # 设置字体大小
    plt.gca().tick_params(axis='both', which='major', labelsize=13)

    # 添加图例
    plt.legend()
    # 保存图表为PNG文件，设置分辨率
    try:
        # 你的图像生成和保存代码
        plt.savefig(fault_diagnosis_image_path)
    except Exception as e:
        print(f"保存图像时出错：{e}")
    # 显示图表

    if bpfi is not None and bpfo is not None and bsf is not None:
        # 确定特征频率以及频率窗口
        characteristic_freqs = {'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf}
        frequency_window = 10  # 特征频率周围的搜索窗口宽度，例如：±10 Hz
        # 创建一个掩码来筛选想要的频率范围和振幅
        mask = (freq >= 60) & (freq <= 300)

        # 只选择0-300 Hz的振幅值和频率值
        freq_filtered = freq[mask]
        am_filtered = am[mask]
        # 寻找峰值
        peaks, _ = find_peaks(am_filtered)

        # 测量峰值的小窗口宽度±3Hz和大窗口宽度±20Hz
        peak_frequency_window = 3
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

        # 假定您有绘图代码
        # plt.plot(...)

        # 生成纯文本格式的报告
        txt_content = f'''
            基于改进小波分解的的海上风机轴承故障诊断结果报告

            时域波形图: 请查阅{latest_image_path}
            故障诊断结果图: 请查阅{fault_diagnosis_image_path}

            结论一: {final_conclusion_1}
            结论二: {final_conclusion_2}

        '''
        # 获取文件的基本名字并去除扩展名
        image_basename = os.path.basename(fault_diagnosis_image_path)
        image_name_without_ext = os.path.splitext(image_basename)[0]

        # 使用图像文件基本名字生成文本报告的唯一文件名
        txt_report_filename = f'{image_name_without_ext}_report.txt'
        folder_path = r'F:\文献'
        txt_report_path = os.path.join(folder_path, txt_report_filename)

        # 将纯文本内容写入 txt 文件
        with open(txt_report_path, 'w', encoding='utf-8') as file:
            file.write(txt_content)

        txt_report_path
        print("txt文本报告已经生成。")
    else:
        print("一个或多个特征频率值为空，跳过故障检测和报告生成流程。")


    return emd_image_path, fault_diagnosis_image_path
