# Description: 该文件包含了一些用于计算图像梯度、边缘检测和法线估计的函数

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cosine

def detect_edges_and_normals(image_path, num_points, sigma=2, margin=30):
    # 1. 读取图像并转换为灰度图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. 对图像进行高斯平滑处理以减少噪声
    smoothed_image = gaussian_filter(image, sigma)

    # 3. 使用 Canny 算子进行边缘检测
    edges = cv2.Canny(smoothed_image, 50, 150)

    # 4. 计算图像梯度的幅度和方向
    sobelx = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)  # x方向的梯度
    sobely = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)  # y方向的梯度

    # 计算梯度方向
    gradient_direction = np.arctan2(sobely, sobelx)

    # 找到边缘像素的位置
    edge_points = np.argwhere(edges > 0)

    # 过滤掉靠近边缘的点，以避免法线超出图像边缘
    filtered_points = [point for point in edge_points if margin <= point[0] < image.shape[0] - margin and margin <= point[1] < image.shape[1] - margin]
    filtered_points = np.array(filtered_points)

    # 随机选择一些边缘点来可视化
    selected_points = filtered_points[np.random.choice(filtered_points.shape[0], num_points, replace=False)]

    return selected_points, gradient_direction, smoothed_image, image

def extract_normal_intensities(selected_points, gradient_direction, image, length=20):
    normal_intensities = []

    for point in selected_points:
        y, x = point
        # 获取梯度方向
        theta = gradient_direction[y, x]

        # 计算法线方向（与梯度方向一致）
        normal_direction = theta

        # 在法线方向上提取灰度值
        line_x = np.linspace(x - length * np.cos(normal_direction), x + length * np.cos(normal_direction), num=2 * length)
        line_y = np.linspace(y - length * np.sin(normal_direction), y + length * np.sin(normal_direction), num=2 * length)

        # 确保坐标在图像范围内
        line_x = np.clip(line_x, 0, image.shape[1] - 1).astype(int)
        line_y = np.clip(line_y, 0, image.shape[0] - 1).astype(int)

        # 获取沿法线方向的灰度值
        intensities = image[line_y, line_x]
        normal_intensities.append(intensities)

    return normal_intensities

def average_normal_intensities(normal_intensities):
    # 对所有法线上的灰度值进行平均
    max_length = max(len(intensities) for intensities in normal_intensities)
    averaged_intensities = np.zeros(max_length)
    count = np.zeros(max_length)

    for intensities in normal_intensities:
        length = len(intensities)
        averaged_intensities[:length] += intensities
        count[:length] += 1

    # 避免除以零的情况
    count[count == 0] = 1
    averaged_intensities /= count

    return averaged_intensities

def calculate_intensity_similarity(normal_intensities):
    num_distributions = len(normal_intensities)
    similarity_matrix = np.zeros((num_distributions, num_distributions))

    for i in range(num_distributions):
        for j in range(num_distributions):
            if i == j:
                similarity_matrix[i, j] = 1.0  # 自己与自己的相似度为 1
            else:
                # 检查向量范数是否为零，避免除以零
                if np.linalg.norm(normal_intensities[i]) == 0 or np.linalg.norm(normal_intensities[j]) == 0:
                    similarity_matrix[i, j] = 0.0  # 设为 0 或者设为一个合理的默认值
                else:
                    similarity = 1 - cosine(normal_intensities[i], normal_intensities[j])
                    similarity_matrix[i, j] = similarity

    return similarity_matrix

def filter_outliers_by_gray_value(normal_intensities):
    # 计算每条法线灰度值的初始值和终止值
    initial_values = [intensities[0] for intensities in normal_intensities]
    terminal_values = [intensities[-1] for intensities in normal_intensities]
    median_initial = np.median(initial_values)
    median_terminal = np.median(terminal_values)

    # 设定阈值，剔除初始值大于中位数或终止值小于中位数的分布
    filtered_indices = [
        i for i, (initial, terminal) in enumerate(zip(initial_values, terminal_values))
        if initial <= median_initial and terminal >= median_terminal
    ]
    filtered_intensities = [normal_intensities[i] for i in filtered_indices]
    
    print(f"初始剔除后保留的分布数量: {len(filtered_intensities)} / {len(normal_intensities)}")
    return filtered_intensities, filtered_indices

