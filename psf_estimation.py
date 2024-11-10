import numpy as np
import math
from numpy import log
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian(x, amplitude, mean, sigma):
    """一维高斯函数"""
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

def estimate_psf(averaged_intensities):
    """
    改进的 PSF 估计函数
    """
    # 1. 平滑 ESF 减少噪声
    esf_smooth = gaussian_filter(averaged_intensities, sigma=1)
    
    # 2. 计算 LSF（处理边界条件）
    lsf = np.gradient(esf_smooth)
    lsf = np.maximum(lsf, 0)  # 确保非负
    
    # 3. 拟合高斯函数（添加错误处理）
    x = np.arange(len(lsf))
    try:
        p0 = [np.max(lsf), len(lsf)/2, len(lsf)/10]
        popt, _ = curve_fit(gaussian, x, lsf, p0=p0, maxfev=2000)
        amplitude, mean, sigma = popt
    except RuntimeError:
        print("高斯拟合失败，使用默认参数")
        amplitude, mean, sigma = np.max(lsf), len(lsf)/2, len(lsf)/10
    
    # 4. 生成 2D PSF（防止数值问题）
    gaussian_lsf = gaussian(x, amplitude, mean, sigma)
    
    # 计算 PSF 大小（添加边界检查）
    psfN = max(3, int(np.ceil(sigma * 4)))  # 至少 3x3
    N = min(psfN * 2 + 1, 31)  # 最大 31x31
    
    # 生成 2D PSF
    x = np.linspace(-N//2, N//2, N)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    psf_2d = np.exp(-R**2 / (2 * sigma**2))
    
    # 确保归一化且非负
    psf_2d = np.maximum(psf_2d, 1e-10)
    psf_2d = psf_2d / psf_2d.sum()
    
    print(f"PSF shape: {psf_2d.shape}")
    print(f"PSF sum: {psf_2d.sum():.6f}")
    
    return lsf, gaussian_lsf, psf_2d
