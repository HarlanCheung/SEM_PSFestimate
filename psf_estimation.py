import numpy as np

def estimate_psf_from_average_intensity(averaged_intensities):
    """
    使用 knife edge 方法从平均灰度分布推测点扩散函数 (PSF)。

    Parameters:
    - averaged_intensities: np.array, 经过滤波的平均灰度分布。

    Returns:
    - LSF_normalized: np.array, 归一化后的 PSF 近似值。
    """
    # 1. 计算边缘扩展函数（ESF）的导数以获得线扩展函数（LSF）
    LSF = np.diff(averaged_intensities)

    # 2. 将 LSF 归一化以获得 PSF 的近似
    LSF_normalized = LSF / np.max(LSF)

    return LSF_normalized
