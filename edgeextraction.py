import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cosine
import seaborn as sns

def detect_edges_and_normals(image_path, sigma=2, num_points=100, margin=30):
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
    # 计算每条法线灰度值的初始值
    initial_values = [intensities[0] for intensities in normal_intensities]
    median_initial = np.median(initial_values)
    
    # 设定阈值，剔除初始值大于中位数的分布
    filtered_indices = [i for i, value in enumerate(initial_values) if value <= median_initial]
    filtered_intensities = [normal_intensities[i] for i in filtered_indices]
    
    print(f"初始剔除后保留的分布数量: {len(filtered_intensities)} / {len(normal_intensities)}")
    return filtered_intensities, filtered_indices

def visualize_results(title, selected_points, gradient_direction, image, normal_intensities, averaged_intensities, similarity_matrix, axes, length=20):
    # 子图 1：显示边缘点和法线
    ax1 = axes[0]
    ax1.imshow(image, cmap='gray')
    ax1.scatter(selected_points[:, 1], selected_points[:, 0], color='red', s=10, label='Selected Edge Points')

    for idx, point in enumerate(selected_points):
        y, x = point
        # 提取正确的梯度方向值
        theta = gradient_direction[idx] if gradient_direction.ndim == 1 else gradient_direction[y, x]
        
        # 计算法线方向的两端点坐标
        x1 = int(x - length * np.cos(theta))
        y1 = int(y - length * np.sin(theta))
        x2 = int(x + length * np.cos(theta))
        y2 = int(y + length * np.sin(theta))

        # 在图中绘制法线
        ax1.plot([x1, x2], [y1, y2], color='blue', linewidth=1)

    ax1.set_title(f'{title}: Edge Points and Normals')
    ax1.legend()

    # 子图 2：显示所有法线的灰度值分布
    ax2 = axes[1]
    for i, intensities in enumerate(normal_intensities):
        ax2.plot(intensities, label=f'Edge Point {i+1}')
    ax2.set_title(f'{title}: Gray Level Changes Along Normals')
    ax2.set_xlabel('Distance along normal direction')
    ax2.set_ylabel('Gray Level')

    # 子图 3：显示平均后的灰度值分布
    ax3 = axes[2]
    ax3.plot(averaged_intensities, label='Averaged Gray Level', color='black', linewidth=2)
    ax3.set_title(f'{title}: Averaged Gray Level Changes')
    ax3.set_xlabel('Distance along normal direction')
    ax3.set_ylabel('Gray Level')
    ax3.legend()

    # 子图 4：显示相似度热图
    ax4 = axes[3]
    sns.heatmap(similarity_matrix, annot=False, cmap='viridis', fmt='.2f', ax=ax4)
    ax4.set_title(f'{title}: Similarity Heatmap')
    ax4.set_xlabel('Distribution Index')
    ax4.set_ylabel('Distribution Index')



if __name__ == "__main__":
    image_path = '/Users/harlan/Documents/shaolab/code_proj/psf/test.tif'
    selected_points, gradient_direction, smoothed_image, image = detect_edges_and_normals(image_path)

    # 提取原图像上的法线灰度值分布
    normal_intensities = extract_normal_intensities(selected_points, gradient_direction, image)
    averaged_intensities = average_normal_intensities(normal_intensities)
    similarity_matrix = calculate_intensity_similarity(normal_intensities)

    # 初步剔除灰度值显著异常的分布
    filtered_intensities, filtered_indices = filter_outliers_by_gray_value(normal_intensities)

    # 更新剔除后的边缘点和梯度方向
    filtered_selected_points = selected_points[filtered_indices]
    filtered_gradient_direction = np.array([gradient_direction[y, x] for y, x in filtered_selected_points])

    # 计算剔除后的平均灰度值分布
    filtered_averaged_intensities = average_normal_intensities(filtered_intensities)
    filtered_similarity_matrix = calculate_intensity_similarity(filtered_intensities)

    # 创建一个新的 figure，包含剔除前后的对比展示
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Comparison Before and After Filtering', fontsize=10)

    # 可视化剔除前的结果
    visualize_results("Before Filtering", selected_points, gradient_direction, image, normal_intensities, averaged_intensities, similarity_matrix, axes[0])

    # 可视化剔除后的结果
    visualize_results("After Filtering", filtered_selected_points, filtered_gradient_direction, image, filtered_intensities, filtered_averaged_intensities, filtered_similarity_matrix, axes[1])

    # 调整布局以减少空白区域
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.3, hspace=0.4)

    # 显示所有图像
    plt.show()
