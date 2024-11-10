from skimage.restoration import richardson_lucy

def deconvolve_image(image, psf, iterations):

    deconvolved_image = richardson_lucy(image, psf, num_iter=iterations)
    
    # 归一化到 [0, 255] 范围
    deconvolved_image = ((deconvolved_image - deconvolved_image.min()) / 
                   (deconvolved_image.max() - deconvolved_image.min()) * 255)
    
    # 添加调试信息
    print(f"Deconvolved image range: [{deconvolved_image.min()}, {deconvolved_image.max()}]")
    print(f"PSF sum: {psf.sum()}")
    
    return deconvolved_image
