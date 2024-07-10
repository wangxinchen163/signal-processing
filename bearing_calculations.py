import numpy as np
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