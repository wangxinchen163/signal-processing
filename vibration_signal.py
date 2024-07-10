import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.signal import hanning, spectrogram
from scipy.fft import fft, ifft, fftshift
from scipy.signal import get_window
import glob
import os
import pandas as pd

config = {
    "font.family": 'serif', # 衬线字体
    "font.size": 10, # 相当于小四大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)


# 假定一个全局集合来跟踪已处理的文件
processed_files = set()
# 新增一个全局变量来存储最后一次处理的文件信息
last_processed_info = (None, None)  # 初始化为 (None, None)

def find_new_csv_files(folder_path, processed_files):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    new_files = [file for file in csv_files if file not in processed_files]
    new_files.sort(key=os.path.getmtime)  # 按文件修改时间排序
    # 打印新文件列表
    print(processed_files)
    print("新文件列表：")
    for file in new_files:
        print(file)
    return new_files


def process_csv_file(fs, csv_file, image_folder_path):
    global last_processed_info  # 使用 global 声明，以便更新全局变量

    csv_basename = os.path.basename(csv_file)
    csv_name_without_ext = os.path.splitext(csv_basename)[0]
    image_filename = f"{csv_name_without_ext}_vibration_signal.jpg"
    image_path = os.path.join(image_folder_path, image_filename)

    try:
        # 读取CSV文件数据
        df = pd.read_csv(csv_file)
        xt = df[df.columns[0]].values  # 假设 xt 是处理后的数据
        Ts = 1 / fs
        # 采样点数
        N = len(xt)
        # 去均值，单位方差
        xt = (xt - np.mean(xt)) / np.std(xt)
        # 绘制图像
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(N) * Ts, xt, color=[147/255, 0/255, 255/255])
        plt.title(f"Vibration Signal of {csv_name_without_ext}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        # 保存图像到指定路径
        plt.savefig(image_path)
        plt.close()



        # 在这里更新 last_processed_info
        last_processed_info = (image_path, xt)  # 更新最后处理的文件信息
        # 将文件名添加到已处理文件的集合中
        processed_files.add(csv_file)  # 正确添加文件到已处理集合
        return image_path, xt
    except Exception as e:
        print(f"处理文件 {csv_file} 时出现错误：{e}")
        return None, None

def vibration(fs, folder_path):
    global last_processed_info  # 使用 global 声明
    image_folder_path = os.path.join(folder_path, 'images')
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

    new_files = find_new_csv_files(folder_path, processed_files)
    if not new_files:
        print("没有新的 CSV 文件需要处理。")
        # 直接返回 last_processed_info，无需再次检查其有效性
        return last_processed_info

    for csv_file in new_files:
        image_path, xt_data = process_csv_file(fs, csv_file, image_folder_path)
        # 更新 last_processed_info 为最新处理的文件信息，无论是否成功
        if image_path and xt_data is not None:
            last_processed_info = (image_path, xt_data)
    plt.close('all')
    # 最后，返回 last_processed_info 而不是 None, None
    # 这里不需要再做检查，因为即使在本次没有新文件，也应该返回最后一次处理的信息
    return last_processed_info
