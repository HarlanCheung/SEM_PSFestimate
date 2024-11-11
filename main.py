import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile as tiff
from functions import (detect_edges_and_normals, extract_normal_intensities, average_normal_intensities, 
                       calculate_emd_similarity_matrix, filter_outliers_by_gray_value)
#from psf_estimation import estimate_psf_from_average_intensity, fit_gaussian_to_psf, expand_psf_to_2d
from psf_estimation import estimate_psf
from RLdeconvolution import deconvolve_image

if __name__ == "__main__":
    image_path = '/Users/harlan/Documents/shaolab/code_proj/psf/test.tif'
    originimage = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    print(originimage.shape)
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    headname = file_name + '_'
    num_points = 200
    selected_points, gradient_direction, smoothed_image, image = detect_edges_and_normals(image_path, num_points)
    '''
    select points & filter & visualization
    '''
    # 提取原图像上的法线灰度值分布
    normal_intensities = extract_normal_intensities(selected_points, gradient_direction, image)
    averaged_intensities = average_normal_intensities(normal_intensities)
    similarity_matrix = calculate_emd_similarity_matrix(normal_intensities)

    # 初步剔除灰度值显著异常的分布
    filtered_intensities, filtered_indices = filter_outliers_by_gray_value(normal_intensities)

    # 更新剔除后的边缘点和梯度方向
    filtered_selected_points = selected_points[filtered_indices]
    filtered_gradient_direction = np.array([gradient_direction[y, x] for y, x in filtered_selected_points])

    # 计算剔除后的平均灰度值分布
    filtered_averaged_intensities = average_normal_intensities(filtered_intensities)
    filtered_similarity_matrix = calculate_emd_similarity_matrix(filtered_intensities)

    # 创建一个新的 figure，包含剔除前后的对比展示
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()  # 将 axes 转换为一维数组以便于索引

    fig.suptitle('Comparison Before and After Filtering', fontsize=10)

    # 可视化剔除前的结果
    # 子图 1：显示边缘点和法线
    ax1 = axes[0]
    ax1.imshow(image, cmap='gray')
    ax1.scatter(selected_points[:, 1], selected_points[:, 0], color='red', s=10, label='Selected Edge Points')

    length = 20
    for idx, point in enumerate(selected_points):
        y, x = point
        theta = gradient_direction[idx] if gradient_direction.ndim == 1 else gradient_direction[y, x]
        x1 = int(x - length * np.cos(theta))
        y1 = int(y - length * np.sin(theta))
        x2 = int(x + length * np.cos(theta))
        y2 = int(y + length * np.sin(theta))
        ax1.plot([x1, x2], [y1, y2], color='blue', linewidth=1)
    ax1.set_title('Before Filtering: Edge Points and Normals')
    ax1.legend()

    # 子图 2：显示所有法线的灰度值分布
    ax2 = axes[1]
    for i, intensities in enumerate(normal_intensities):
        ax2.plot(intensities, label=f'Edge Point {i+1}')
    ax2.set_title('Before Filtering: Gray Level Changes Along Normals')
    ax2.set_xlabel('Distance along normal direction')
    ax2.set_ylabel('Gray Level')

    # 子图 3：显示平均后的灰度值分布
    ax3 = axes[2]
    ax3.plot(averaged_intensities, label='Averaged Gray Level', color='black', linewidth=2)
    ax3.set_title('Before Filtering: Averaged Gray Level Changes')
    ax3.set_xlabel('Distance along normal direction')
    ax3.set_ylabel('Gray Level')
    ax3.legend()

    # 子图 4：显示相似度热图
    ax4 = axes[3]
    import seaborn as sns
    sns.heatmap(similarity_matrix, annot=False, cmap='viridis', fmt='.2f', ax=ax4)
    ax4.set_title('Before Filtering: Similarity Heatmap')
    ax4.set_xlabel('Distribution Index')
    ax4.set_ylabel('Distribution Index')

    # 可视化剔除后的结果
    # 子图 5：显示边缘点和法线
    ax5 = axes[4]
    ax5.imshow(image, cmap='gray')
    ax5.scatter(filtered_selected_points[:, 1], filtered_selected_points[:, 0], color='red', s=10, label='Filtered Edge Points')
    for idx, point in enumerate(filtered_selected_points):
        y, x = point
        theta = filtered_gradient_direction[idx]
        x1 = int(x - length * np.cos(theta))
        y1 = int(y - length * np.sin(theta))
        x2 = int(x + length * np.cos(theta))
        y2 = int(y + length * np.sin(theta))
        ax5.plot([x1, x2], [y1, y2], color='blue', linewidth=1)
    ax5.set_title('After Filtering: Edge Points and Normals')
    ax5.legend()

    # 子图 6：显示剔除后的法线灰度值分布
    ax6 = axes[5]
    for i, intensities in enumerate(filtered_intensities):
        ax6.plot(intensities, label=f'Filtered Edge Point {i+1}')
    ax6.set_title('After Filtering: Gray Level Changes Along Normals')
    ax6.set_xlabel('Distance along normal direction')
    ax6.set_ylabel('Gray Level')

    # 子图 7：显示剔除后的平均灰度值分布
    ax7 = axes[6]
    ax7.plot(filtered_averaged_intensities, label='Filtered Averaged Gray Level', color='black', linewidth=2)
    ax7.set_title('After Filtering: Averaged Gray Level Changes')
    ax7.set_xlabel('Distance along normal direction')
    ax7.set_ylabel('Gray Level')
    ax7.legend()

    # 子图 8：显示剔除后的相似度热图
    ax8 = axes[7]
    sns.heatmap(filtered_similarity_matrix, annot=False, cmap='viridis', fmt='.2f', ax=ax8)
    ax8.set_title('After Filtering: Similarity Heatmap')
    ax8.set_xlabel('Distribution Index')
    ax8.set_ylabel('Distribution Index')

    # 调整布局以减少空白区域
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.3, hspace=0.4)
    
    # 保存结果
    plt.savefig(headname + 'edge_extraction&graysale_distribution.png')
    
    # 显示所有图像
    # plt.show()

    '''
    PSF estiamtion & visualization
    '''

    # 估计PSF
    lsf, gaussian_lsf, psf = estimate_psf(averaged_intensities)

    # 可视化原始 PSF 和拟合的高斯曲线

    x_data = np.arange(len(lsf))
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, lsf, label='Estimated PSF (using Knife Edge Method)', linestyle='-', marker='o')
    plt.plot(x_data, gaussian_lsf, label='Fitted Gaussian', color='red', linestyle='--')
    plt.xlabel('Position along normal direction')
    plt.ylabel('Normalized Intensity')
    plt.title('Estimated PSF and Fitted Gaussian')
    plt.legend()
    plt.grid()
    plt.savefig(headname + 'psf_fitted_gaussian.png')

    # 可视化PSF
    plt.figure(figsize=(10, 8))
    plt.imshow(psf, cmap='jet')
    plt.colorbar()
    plt.title('Estimated 2D PSF')
    plt.savefig(headname + 'estimated_psf.png')

    # Perform deconvolution

    deconvolved_image = deconvolve_image(originimage, psf, 2)

    print(deconvolved_image.shape)

    # Ensure the deconvolved image is in the correct format
    deconvolved_image = np.clip(deconvolved_image, 0, 255).astype(np.uint8)

    # Save and display the deconvolved image
    tiff.imwrite(headname + 'deconvolved_image.tif', deconvolved_image)
    plt.imshow(deconvolved_image, cmap='gray')
    plt.title('Deconvolved Image')
    plt.axis('off')
    #plt.show()