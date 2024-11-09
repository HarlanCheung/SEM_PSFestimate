import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def estimate_psf_from_average_intensity(averaged_intensities, sigma=2):
    """
    使用 knife edge 方法从平均灰度分布推测点扩散函数 (PSF)，并对 ESF 进行平滑处理。

    Parameters:
    - averaged_intensities: np.array, 经过滤波的平均灰度分布。
    - sigma: float, 高斯平滑的标准差。

    Returns:
    - LSF_normalized: np.array, 归一化后的 PSF 近似值。
    """
    # 1. 对 ESF 进行高斯平滑处理，以减少噪声
    smoothed_esf = gaussian_filter(averaged_intensities, sigma=sigma)

    # 2. 计算边缘扩展函数（ESF）的导数以获得线扩展函数（LSF）
    LSF = np.diff(smoothed_esf)

    # 3. 将 LSF 归一化以获得 PSF 的近似
    LSF_normalized = LSF / np.max(LSF)

    return LSF_normalized

def gaussian(x, a, mu, sigma):
    """
    定义高斯函数，用于拟合 PSF。

    Parameters:
    - x: np.array, 自变量。
    - a: float, 振幅。
    - mu: float, 均值。
    - sigma: float, 标准差。

    Returns:
    - np.array, 高斯函数的值。
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def fit_gaussian_to_psf(psf):
    """
    使用高斯函数拟合 PSF。

    Parameters:
    - psf: np.array, 归一化后的 PSF。

    Returns:
    - fitted_gaussian: np.array, 拟合的高斯函数值。
    - params: list, 高斯拟合参数 [a, mu, sigma]。
    """
    # 定义 x 轴数据
    x_data = np.arange(len(psf))
    y_data = psf

    # 初始猜测参数 (幅度，均值，标准差)
    initial_guess = [1, len(x_data) / 2, 5]

    # 使用 curve_fit 进行拟合
    params, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
    
    # 拟合得到的参数
    a, mu, sigma = params

    # 生成拟合的高斯曲线
    fitted_gaussian = gaussian(x_data, a, mu, sigma)
    return fitted_gaussian, params

def expand_psf_to_2d(fitted_gaussian, size=100):
    """
    将一维高斯拟合后的 PSF 扩展为二维并进行三维可视化。

    Parameters:
    - fitted_gaussian: np.array, 拟合的高斯曲线。
    - size: int, 二维图像的大小（默认 100x100）。
    """
    # 创建二维高斯 PSF
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    
    # 使用拟合得到的 sigma 计算二维高斯
    sigma = len(fitted_gaussian) / (2 * np.sqrt(2 * np.log(2)))
    gaussian_2d = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # 归一化二维高斯 PSF
    gaussian_2d_normalized = gaussian_2d / np.max(gaussian_2d)

    # 打印高斯函数的标准差 sigma
    print(f"Sigma of the fitted Gaussian: {sigma}")

    return gaussian_2d_normalized

