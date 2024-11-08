import matplotlib.pyplot as plt
import numpy as np
from edgeextraction import (detect_edges_and_normals, extract_normal_intensities, average_normal_intensities, 
                       calculate_intensity_similarity, filter_outliers_by_gray_value, visualize_results)

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
    
    # 保存结果
    plt.savefig('edge_extraction&graysale_distribution.png')
    
    # 显示所有图像
    plt.show()
