from vibration_signal import vibration
from xiaobo_feijie import xiaobo_feijie_acq
from emd_feijie import emd_feijie_acq
from SK_acq import sk_signal
from matplotlib import rcParams
import os
import itertools

from PIL import Image, ImageTk, ImageSequence
import tkinter as tk
from contextlib import redirect_stdout
import io
import sys
from tkinter import PhotoImage
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from bearing_calculations import bearing_fault_freq_cal
from SK3 import process_and_filter_signal
import tkinter as tk
from tkinter import messagebox  # 确保包含这一行
import numpy as np
from tkinter import Tk, StringVar, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from data_acq import data_acquision
import matplotlib as mpl

# 设置agg.path.chunksize
mpl.rcParams['agg.path.chunksize'] = 10000  # 或者更大的值，根据需要调整

config = {
    "font.family": 'serif', # 衬线字体
    "font.size": 10, # 相当于小四大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)
# 控制循环是否继续的变量
is_running = False

def auto_update_image():
    if is_running:
        # 检查新图像并更新
        update_image_display()
        # 在图像更新后运行主振动分析
        main_vibration_analysis()
        main_wavelet_analysis()
        main_emd_analysis()
        # 每隔一定时间（例如5000毫秒）再次调用此函数，形成循环
        root.after(180000, auto_update_image)
def start_loop():
    global is_running
    is_running = True
    auto_update_image()

def stop_loop():
    global is_running
    is_running = False

def update_image_display():
    folder_path = r'F:\文献'
    fs = float(param_vars["fs"].get())  # 将字符串值转换为浮点数
    latest_image_path, _ = vibration(fs, folder_path)
    print(f"最新图像路径：{latest_image_path}")
    # 调用vibration函数获取最新图像路径和xt数据，但忽略xt数据

    if latest_image_path and os.path.exists(latest_image_path):
        try:
            img = Image.open(latest_image_path)  # 尝试打开图像
            img_resized = img.resize((400, 350), Image.ANTIALIAS)  # 调整图像大小以适应GUI
            img_tk = ImageTk.PhotoImage(img_resized)  # 直接从调整大小后的图像创建PhotoImage

            # 如果之前已经显示了图像，则先移除
            for widget in image_frame.winfo_children():
                widget.destroy()

            # 显示最新图像
            image_label = tk.Label(image_frame, image=img_tk)
            image_label.image = img_tk  # 保持对图像的引用，防止被垃圾回收
            image_label.pack()
        except Exception as e:
            print(f"加载或处理图像时出错：{e}")
            print(f"图像路径：{latest_image_path}")

    else:
        print(f"无法找到或打开图像：{latest_image_path}")
def update_harmonic_display(spectrum_image_path):
    print(f"调试信息：尝试更新谐波滤除模块显示。给定图像路径：{spectrum_image_path}")
    global img_tk_reference  # 假设有一个全局变量来保持图像引用
    print(f"调试信息：尝试更新谐波滤除模块显示。给定图像路径：{spectrum_image_path}")
    if spectrum_image_path and os.path.exists(spectrum_image_path):
        try:
            img = Image.open(spectrum_image_path)
            print("调试信息：图像加载成功。")
            img = img.resize((400, 350), Image.ANTIALIAS)
            print("调试信息：图像调整大小成功。")

            img_tk = ImageTk.PhotoImage(img)
            img_tk_reference = img_tk  # 更新全局引用
            print("调试信息：创建PhotoImage成功。")

            for widget in harmonic_frame.winfo_children():
                widget.destroy()
            print("调试信息：已清除谐波滤除模块中的旧图像。")

            image_label = tk.Label(harmonic_frame, image=img_tk)
            image_label.image = img_tk  # 再次确保引用保持
            image_label.pack()
            print("调试信息：新图像已成功显示。")
        except Exception as e:
            print(f"更新谐波滤除模块显示时出错：{e}")
    else:
        print("没有提供有效的图像路径，或路径不指向一个存在的文件，无法更新谐波滤除模块。")

def update_wavelet_display(wavelet_image_path):

    global img_tk_reference_wavelet  # 假设有一个全局变量来保持图像引用
    print(f"调试信息：尝试更新小波分析模块显示。给定图像路径：{wavelet_image_path}")
    if wavelet_image_path and os.path.exists(wavelet_image_path):
        try:
            img = Image.open(wavelet_image_path)
            print("调试信息：图像加载成功。")
            img = img.resize((400, 520), Image.ANTIALIAS)  # 根据需要调整大小
            print("调试信息：图像调整大小成功。")

            img_tk_wavelet = ImageTk.PhotoImage(img)
            img_tk_reference_wavelet = img_tk_wavelet  # 更新全局引用
            print("调试信息：创建PhotoImage成功。")

            for widget in wavelet_frame.winfo_children():
                widget.destroy()
            print("调试信息：已清除小波分析模块中的旧图像。")

            image_label = tk.Label(wavelet_frame, image=img_tk_wavelet)
            image_label.image = img_tk_wavelet  # 再次确保引用保持
            image_label.pack()
            print("调试信息：新图像已成功显示。")
        except Exception as e:
            print(f"更新小波分析模块显示时出错：{e}")
    else:
        print("没有提供有效的图像路径，或路径不指向一个存在的文件，无法更新小波分析模块。")
def update_emd_display(emd_image_path):
    global img_tk_reference_emd  # 假设有一个全局变量来保持图像引用，避免垃圾回收
    print(f"调试信息：尝试更新EMD分析模块显示。给定图像路径：{emd_image_path}")

    if emd_image_path and os.path.exists(emd_image_path):
        try:
            img = Image.open(emd_image_path)
            print("调试信息：图像加载成功。")
            img = img.resize((400, 520), Image.ANTIALIAS)  # 根据需要调整大小
            print("调试信息：图像调整大小成功。")

            img_tk_emd = ImageTk.PhotoImage(img)
            img_tk_reference_emd = img_tk_emd  # 更新全局引用
            print("调试信息：创建PhotoImage成功。")

            for widget in emd_frame.winfo_children():
                widget.destroy()
            print("调试信息：已清除EMD分析模块中的旧图像。")

            image_label = tk.Label(emd_frame, image=img_tk_emd)
            image_label.image = img_tk_emd  # 再次确保引用保持
            image_label.pack()
            print("调试信息：新图像已成功显示。")
        except Exception as e:
            print(f"更新EMD分析模块显示时出错：{e}")
    else:
        print("没有提供有效的图像路径，或路径不指向一个存在的文件，无法更新EMD分析模块。")
def update_fault_diagnosis_display(fault_diagnosis_image_path):
    global img_tk_reference_fault_diagnosis  # 全局变量来保持图像引用
    print(f"调试信息：尝试更新故障诊断模块显示。给定图像路径：{fault_diagnosis_image_path}")

    if fault_diagnosis_image_path and os.path.exists(fault_diagnosis_image_path):
        try:
            img = Image.open(fault_diagnosis_image_path)
            print("调试信息：图像加载成功。")
            img = img.resize((400, 350), Image.ANTIALIAS)  # 根据需要调整大小
            print("调试信息：图像调整大小成功。")

            img_tk_fault_diagnosis = ImageTk.PhotoImage(img)
            img_tk_reference_fault_diagnosis = img_tk_fault_diagnosis  # 更新全局引用
            print("调试信息：创建PhotoImage成功。")

            # 清除fault_diagnosis_frame中的旧图像或其他小部件
            for widget in fault_diagnosis_frame.winfo_children():
                widget.destroy()
            print("调试信息：已清除故障诊断模块中的旧图像。")

            # 在fault_diagnosis_frame中显示新图像
            image_label = tk.Label(fault_diagnosis_frame, image=img_tk_fault_diagnosis)
            image_label.image = img_tk_fault_diagnosis  # 再次确保引用保持
            image_label.pack()
            print("调试信息：新图像已成功显示。")
        except Exception as e:
            print(f"更新故障诊断模块显示时出错：{e}")
    else:
        print("没有提供有效的图像路径，或路径不指向一个存在的文件，无法更新故障诊断模块。")

# 创建主窗口
root = tk.Tk()
root.configure(background='lightblue')  # 设置窗口背景色
root.title("基于改进小波分解的海上风机轴承故障诊断系统")
def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", False)  # 切换到非全屏模式

# 绑定Esc键与退出全屏模式的函数
root.bind("<Escape>", toggle_fullscreen)

# 设置窗口为全屏模式
root.attributes('-fullscreen', True)

# 加载图片（替换'image_path.png'为你的图片路径）
logo_path = 'F:\文献\zju.png'  # 这里填入你的图片文件路径
image = Image.open(logo_path)
image = image.resize((100, 100), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(image)

# 使用place直接定位logo_label
logo_label = tk.Label(root, image=photo)
logo_label.place(x=100, y=0)
# 第一个Label，添加边框样式和宽度
title_label1 = tk.Label(root, text="海上风机轴承", font=("Microsoft YaHei", 25, "bold"),
                        bg='blue', relief="groove", borderwidth=2, fg="white")
title_label1.place(x=200, y=0)  # x的位置根据实际需要调整，y=0表示位于顶部

# 第二个Label，"故障诊断系统"作为文本，也添加边框样式和宽度
title_label2 = tk.Label(root, text="故障诊断系统", font=("Microsoft YaHei", 25, "bold"),
                        bg='blue', relief="groove", borderwidth=2, fg="white")
title_label2.place(x=200, y=47)  # y的值根据第一个Label的位置和期望的间距调整


# 加载图片（替换'image_path.png'为你的图片路径）
logo_path2 = 'F:\文献\hnu.png'  # 这里填入你的图片文件路径
image2 = Image.open(logo_path2)
image2 = image2.resize((100, 100), Image.ANTIALIAS)
photo2 = ImageTk.PhotoImage(image2)

# 使用place直接定位logo_label
logo_label2 = tk.Label(root, image=photo2)
logo_label2.place(x=0, y=0)
# 假设logo图片的高度为100
logo_height = 100
# 参数设置框架，更新位置为第一行下面
param_frame = tk.LabelFrame(root, text="参数设置", bg='white', width=500, height=400,
                            font=("Microsoft YaHei", 17, "bold"), relief="sunken", borderwidth=13)

param_frame.place(x=0, y=logo_height, width=400, height=350)
param_frame.grid_propagate(False)  # 也可以防止这个框架根据其内容自动调整大小
param_frame.grid_propagate(False)  # 防止框架根据其内容自动调整大小以适应内容
# 假设param_frame和freq_frame之间的间隙为20像素
space_between_frames = 10
# 现在放置freq_frame，确保它的y坐标是param_frame的y坐标加上param_frame的高度加上间隙
freq_frame_y = 420 + space_between_frames  # 假设logo_height是logo图片的高度加上其自身的间距

# 特征频率计算框架
freq_frame = tk.LabelFrame(root, text="故障特征频率计算", bg='white',
                           font=("Helvetica", 17, "bold"), relief="sunken", borderwidth=13)
freq_frame.place(x=0, y=freq_frame_y, width=400, height=400)
freq_frame.grid_columnconfigure(0, weight=1)  # 使列能够扩展，为居中提供空间
freq_frame.grid_propagate(False)  # 阻止框架根据其内容自动调整大小
# 数据采集模块框架
image_frame = tk.LabelFrame(root, text="数据采集模块", font=("Microsoft YaHei", 17, "bold"),
                            borderwidth=2, relief="raised", bg='lightgray')
image_frame.place(x=420, y=0, width=400, height=400)

# 谐波滤除模块框架
harmonic_frame = tk.LabelFrame(root, text="谐波滤除模块", font=("Microsoft YaHei", 17, "bold"),
                            borderwidth=2, relief="raised", bg='lightgray')
harmonic_frame.place(x=860, y=0, width=400, height=400)


# 小波分解模块框架
wavelet_frame = tk.LabelFrame(root, text="小波分解模块", font=("Microsoft YaHei", 17, "bold"),
                            borderwidth=2, relief="raised", bg='lightgray')
wavelet_frame.place(x=420, y=freq_frame_y, width=400, height=520)
wavelet_frame.grid_propagate(False)
wavelet_frame.config(width=400, height=520)

# EMD分解模块框架
emd_frame = tk.LabelFrame(root, text="EMD分解模块", font=("Microsoft YaHei", 17, "bold"),
                            borderwidth=2, relief="raised", bg='lightgray')
emd_frame.place(x=860, y=freq_frame_y, width=400, height=520)
emd_frame.grid_propagate(False)
emd_frame.config(width=400, height=520)
# 故障诊断模块框架
fault_diagnosis_frame = tk.LabelFrame(root, text="故障诊断模块", font=("Microsoft YaHei", 17, "bold"),
                            borderwidth=2, relief="raised", bg='lightgray')
fault_diagnosis_frame.place(x=1310, y=0, width=400, height=400)




# 参数的标签和变量
param_labels = {
    "n": "滚子数量",
    "alpha": "接触角(°)",
    "d": "滚子直径(mm)",
    "D": "轴承节距直径(mm)",
    "fr": "转速(r/min)",
    "fs": "采样频率(Hz)",
    "kthr": "谐波滤除阈值"
}
# 使用StringVar代替直接存储Entry对象
param_vars = {
    "n": tk.StringVar(value="9"),
    "alpha": tk.StringVar(value="0"),
    "d": tk.StringVar(value="9.53"),
    "D": tk.StringVar(value="46.4"),
    "fr": tk.StringVar(value="1449.6519"),
    "fs": tk.StringVar(value="12000"),
    "kthr": tk.StringVar(value="-0.715")
}
# 字体设置，例如：字体类型为Helvetica，大小为12，颜色为蓝色
font_settings = ("Microsoft YaHei", 15)
font_color = "blue"
# 为每个参数创建标签和输入框，并绑定StringVar
# 使用自定义字体和颜色创建标签，以及对应的输入框
for idx, (key, label_text) in enumerate(param_labels.items()):
    # 应用字体设置和颜色
    label = tk.Label(param_frame, text=label_text, font=font_settings, fg=font_color)
    label.grid(row=idx, column=0, sticky="w")

    # 创建输入框，绑定到对应的变量
    entry = tk.Entry(param_frame, textvariable=param_vars[key], font=font_settings, fg=font_color)
    entry.grid(row=idx, column=1, sticky="e")
def main_emd_analysis():
    # 先调用 vibration 函数
    folder_path = r'F:\文献'
    fs = float(param_vars["fs"].get())  # 将字符串值转换为浮点数
    latest_image_path, latest_xt_data = vibration(fs, folder_path)

    # 移除了原先的提前返回逻辑
    # if latest_image_path is None or latest_xt_data is None:
    #     print("没有可用的数据进行后续处理。")
    #     return

    # 后续处理的代码示例
    if latest_image_path is not None:
        # 如果 latest_image_path 不为空，执行相关操作
        print(f"最新图像的路径是: {latest_image_path}")
        # 这里可以是打开图像、显示图像或其他逻辑
    else:
        print("没有生成新的图像，或者图像路径未更新。")

    # 从界面获取采样率和阈值，使用 param_vars 获取参数值
    fs = float(param_vars["fs"].get())  # 将字符串值转换为浮点数
    kthr = float(param_vars["kthr"].get())  # 将字符串值转换为浮点数
    filtered_signal, spectrum_image_path = sk_signal(fs, kthr, latest_xt_data, latest_image_path)
    at1, wavelet_image_path = xiaobo_feijie_acq(fs, filtered_signal, latest_image_path)
    emd_image_path, fault_diagnosis_image_path = emd_feijie_acq(fs, at1, latest_image_path, result_labels)

    if emd_image_path is not None:
        update_emd_display(emd_image_path)
    else:
        print("未能生成小波图像，因此无法更新显示。")
    if fault_diagnosis_image_path is not None:
        update_fault_diagnosis_display(fault_diagnosis_image_path)
    else:
        print("未能生成故障诊断图像，因此无法更新显示。")
def main_wavelet_analysis():
    # 先调用 vibration 函数
    folder_path = r'F:\文献'
    fs = float(param_vars["fs"].get())  # 将字符串值转换为浮点数
    latest_image_path, latest_xt_data = vibration(fs, folder_path)

    # 移除了原先的提前返回逻辑
    # if latest_image_path is None or latest_xt_data is None:
    #     print("没有可用的数据进行后续处理。")
    #     return

    # 后续处理的代码示例
    if latest_image_path is not None:
        # 如果 latest_image_path 不为空，执行相关操作
        print(f"最新图像的路径是: {latest_image_path}")
        # 这里可以是打开图像、显示图像或其他逻辑
    else:
        print("没有生成新的图像，或者图像路径未更新。")

    # 从界面获取采样率和阈值，使用 param_vars 获取参数值
    fs = float(param_vars["fs"].get())  # 将字符串值转换为浮点数
    kthr = float(param_vars["kthr"].get())  # 将字符串值转换为浮点数
    filtered_signal, spectrum_image_path = sk_signal(fs, kthr, latest_xt_data, latest_image_path)
    at1, wavelet_image_path = xiaobo_feijie_acq(fs, filtered_signal, latest_image_path)
    if wavelet_image_path is not None:
        update_wavelet_display(wavelet_image_path)
    else:
        print("未能生成小波图像，因此无法更新显示。")
def main_vibration_analysis():
    # 先调用 vibration 函数
    folder_path = r'F:\文献'
    fs = float(param_vars["fs"].get())  # 将字符串值转换为浮点数
    latest_image_path, latest_xt_data = vibration(fs, folder_path)

    # 移除了原先的提前返回逻辑
    # if latest_image_path is None or latest_xt_data is None:
    #     print("没有可用的数据进行后续处理。")
    #     return

    # 后续处理的代码示例
    if latest_image_path is not None:
        # 如果 latest_image_path 不为空，执行相关操作
        print(f"最新图像的路径是: {latest_image_path}")
        # 这里可以是打开图像、显示图像或其他逻辑
    else:
        print("没有生成新的图像，或者图像路径未更新。")

    if latest_xt_data is not None:
        # 如果 latest_xt_data 不为空，执行相关操作
        print("最新的振动数据已加载，可以进行分析。")
        # 这里可以是数据分析、绘图或其他逻辑
    else:
        print("没有加载新的振动数据，或者数据未更新。")

    # 从界面获取采样率和阈值，使用 param_vars 获取参数值
    fs = float(param_vars["fs"].get())  # 将字符串值转换为浮点数
    kthr = float(param_vars["kthr"].get())  # 将字符串值转换为浮点数
    Data = latest_xt_data

    # 然后使用 vibration 函数的输出调用 sk_signal 函数
    # 假设 sk_signal 函数已经修改，以返回图像路径
    filtered_signal, spectrum_image_path = sk_signal(fs, kthr, latest_xt_data, latest_image_path)

    # 检查 spectrum_image_path 是否有效

    if spectrum_image_path is not None:
        update_harmonic_display(spectrum_image_path)
    else:
        print("未能生成图像，因此无法更新显示。")





def calculate_freqs():
    # 从输入中获取参数
    try:
        # 使用 param_vars 获取参数值
        n_val = int(param_vars["n"].get())
        alpha_val = float(param_vars["alpha"].get()) * np.pi / 180  # 转换为弧度
        d_val = float(param_vars["d"].get())
        D_val = float(param_vars["D"].get())
        fr_val = float(param_vars["fr"].get())

        # 执行计算
        bpfi, bpfo, bsf, ftf, _ = bearing_fault_freq_cal(n_val, d_val, D_val, alpha_val, fr_val)
        # 假设你想要的字体大小和样式
        font_settings = ("Helvetica", 15)  # 例如，字体大小为12
        # 更新标签显示结果
        result_labels["bpfi"].config(text=f"{bpfi:.2f} Hz", font=font_settings)
        result_labels["bpfo"].config(text=f"{bpfo:.2f} Hz", font=font_settings)
        result_labels["bsf"].config(text=f"{bsf:.2f} Hz", font=font_settings)
        result_labels["ftf"].config(text=f"{ftf:.2f} Hz", font=font_settings)
    except ValueError as e:
        messagebox.showerror("错误", f"输入的参数无效: {e}")
style = ttk.Style()
# 配置全局的TLabel样式以改变字体大小
style.configure('TLabel', font=('Helvetica', 15))
# 创建特征频率结果的标签
result_labels = {}
for idx, label in enumerate(["bpfi", "bpfo", "bsf", "ftf"]):
    ttk.Label(freq_frame, text=f"{label.upper()}:").grid(row=idx, column=0, sticky="w")
    result_labels[label] = ttk.Label(freq_frame, text="N/A")
    result_labels[label].grid(row=idx, column=1, sticky="w")


# 注意，这里的背景颜色设置可能不会影响所有平台的 ttk.Button
# 为按钮定义自定义样式
style.configure("Red.TButton", foreground="white", background="red", font=('Helvetica', 12))
style.configure("Green.TButton", foreground="white", background="green", font=('Helvetica', 12))
style.configure("Blue.TButton", foreground="white", background="blue", font=('Helvetica', 12))

calc_button = tk.Button(freq_frame, text="计算特征频率", command=calculate_freqs, fg="white", bg="blue", font=("Helvetica", 15, "bold"), relief="raised", borderwidth=6)
calc_button.grid(row=len(param_vars), column=0, columnspan=2, pady=5, sticky="ew")

start_button = tk.Button(freq_frame, text="开始", command=start_loop, fg="white", bg="red", font=("Helvetica", 15, "bold"), relief="raised", borderwidth=6)
start_button.grid(row=len(param_vars)+1, column=0, pady=5, sticky="ew")

stop_button = tk.Button(freq_frame, text="停止", command=stop_loop, fg="white", bg="green", font=("Helvetica", 15, "bold"), relief="raised", borderwidth=6)
stop_button.grid(row=len(param_vars)+1, column=1, pady=5, sticky="ew")


# 使用grid布局管理器来放置开始和结束按钮
# 注意，这里的row参数需要根据前面按钮和组件的数量适当调整
start_button.grid(row=len(param_vars) + 1, column=0, columnspan=2, pady=5, sticky="ew")  # 放在“计算特征频率”按钮的下面
stop_button.grid(row=len(param_vars) + 2, column=0, columnspan=2, pady=5, sticky="ew")  # 放在“开始”按钮的下面
# 使用image_frame作为参数调用display_image_on_gui
# 启动自动更新机制
auto_update_image()
gif_path = "F:/文献/fngji.gif"  # 确保使用正确的路径
gif = Image.open(gif_path)
# 将GIF的每一帧存储到列表中
frames = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(gif)]

# 创建用于显示GIF帧的Label
label = tk.Label(root)
label.place(x=1310, y=420)

# 创建迭代器循环遍历GIF的每一帧
frames_cycle = itertools.cycle(frames)

def update_frame():
    frame = next(frames_cycle)
    label.config(image=frame)
    root.after(100, update_frame)

# 启动动画
update_frame()
# 启动主事件循环
root.mainloop()