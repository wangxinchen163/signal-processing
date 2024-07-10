import scipy.io as scio
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