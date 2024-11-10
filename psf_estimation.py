import numpy as np
import math
from numpy import log
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian(x, amplitude, mean, sigma):
    """一维高斯函数"""
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

def estimate_psf(edge_spread_function):
    """
    从边缘扩散函数(ESF)估计点扩散函数(PSF)
    
    Parameters:
        edge_spread_function: 边缘扩散函数数据
        window_size: 输出PSF的大小
    
    Returns:
        psf_2d: 二维点扩散函数
    """
    # 1. 平滑ESF减少噪声
    esf_smooth = gaussian_filter(edge_spread_function, sigma=1)
    
    # 2. 计算LSF（ESF的导数）
    lsf = np.gradient(esf_smooth)
    
    # 3. 拟合高斯函数
    x = np.arange(len(lsf))
    try:
        # 初始参数猜测：[振幅, 均值, 标准差]
        p0 = [np.max(lsf), len(lsf)/2, len(lsf)/10]
        popt, _ = curve_fit(gaussian, x, lsf, p0=p0)
        amplitude, mean, sigma = popt
    except RuntimeError:
        print("高斯拟合失败，使用默认参数")
        amplitude, mean, sigma = 1.0, len(lsf)/2, len(lsf)/10
    
    gaussian_lsf = gaussian(x, amplitude, mean, sigma)
    
    #计算PSF 大小
    psfN = np.ceil(sigma / math.sqrt(8 * log(2)) * math.sqrt(-2 * log(0.0002))) + 1
    N = psfN * 2 + 1

    # 4. 生成2D PSF
    x = np.arange(-np.fix((N / 2)), np.ceil((N / 2)),dtype='float32')
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    
    # 使用拟合得到的sigma生成2D高斯
    R = np.sqrt(X**2 + Y**2)
    psf_2d = np.exp(-R**2 / (2 * sigma**2))
    
    # 归一化，确保和为1
    psf_2d = psf_2d / psf_2d.sum()

    
    # 打印调试信息
    print(f"高斯拟合参数: 振幅={amplitude:.2f}, 均值={mean:.2f}, sigma={sigma:.2f}")
    print(f"PSF shape: {psf_2d.shape}")
    print(f"PSF sum: {psf_2d.sum():.6f}")
    
    return lsf, gaussian_lsf, psf_2d
