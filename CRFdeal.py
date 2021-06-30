import numpy as np
import pydensecrf.densecrf as dcrf
import cv2
try:
    from cv2 import imread, imwrite
except ImportError:
    # 如果没有安装OpenCV，就是用skimage
    from skimage.io import imread, imsave
    imwrite = imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

"""
original_image_path  原始图像路径
predicted_image_path  之前用自己的模型预测的图像路径
CRF_image_path  即将进行CRF后处理得到的结果图像保存路径
"""

# def CRFs(original_image_path, predicted_image_path, CRF_image_path):
def CRFs(img, anno_lbl):
    # img = imread(original_image_path)

    # 将predicted_image的RGB颜色转换为uint32颜色 0xbbggrr
    # anno_lbl = imread(predicted_image_path,cv2.IMREAD_GRAYSCALE).astype(np.uint32)
    # anno_lbl = anno_lbl + anno_lbl + anno_lbl
    # print(anno_lbl.shape)
    # 将uint32颜色转换为1,2,...
    # colors, labels = np.unique(anno_lbl, return_inverse=True)
    labels=anno_lbl//255
    # print(labels.shape)
    # 如果你的predicted_image里的黑色（0值）不是待分类类别，表示不确定区域，即将分为其他类别
    # 那么就取消注释以下代码
    # HAS_UNK = 0 in colors
    # if HAS_UNK:
    # colors = colors[1:]

    # 创建从predicted_image到32位整数颜色的映射。
    # colorize = np.empty((len(colors), 3), np.uint8)
    # colorize[:, 0] = (colors & 0x0000FF)
    # colorize[:, 1] = (colors & 0x00FF00) >> 8
    # colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 计算predicted_image中的类数。
    n_labels = len(set(labels.flat))
    # print(n_labels)
    # n_labels = len(set(labels.flat)) - int(HAS_UNK) ##如果有不确定区域，用这一行代码替换上一行

    ###########################
    ###     设置CRF模型     ###
    ###########################
    use_2d = False
    # use_2d = True
    ###########################################################
    ##不是很清楚什么情况用2D
    ##作者说“对于图像，使用此库的最简单方法是使用DenseCRF2D类”
    ##作者还说“DenseCRF类可用于通用（非二维）密集CRF”
    ##但是根据我的测试结果一般情况用DenseCRF比较对
    #########################################################33
    if use_2d:
        # 使用densecrf2d类
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 增加了与颜色无关的术语，功能只是位置而已
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 增加了颜色相关术语，即特征是(x,y,r,g,b)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        # 使用densecrf类
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 这将创建与颜色无关的功能，然后将它们添加到CRF中
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 这将创建与颜色相关的功能，然后将它们添加到CRF中
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ###         做推理和计算         ###
    ####################################

    # 进行5次推理
    Q = d.inference(5)

    # 找出每个像素最可能的类
    MAP = np.argmax(Q, axis=0)

    # 将predicted_image转换回相应的颜色并保存图像
    # MAP = colorize[MAP, :]
    MAP=MAP*255
    return MAP.reshape((img.shape[0],img.shape[1]))
    imwrite(CRF_image_path, MAP.reshape((img.shape[0],img.shape[1])))
    print("CRF图像保存在", CRF_image_path, "!")
# CRFs(r'G:\Remote_sensing_extract\H48F019017\Cut_0.tif',r'G:\Remote_sensing_extract\H48F019017\Unet\Cut_0.tif',r'G:\Remote_sensing_extract\H48F019017\Unet\Cut_0_CRF.png')
# # """2类 crf"""
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# def dense_crf_2d(img, output_probs): # img 为H，*W*C 的原图，output_probs 为 输出概率 sigmoid 输出（h，w），#seg_map - 假设为语义分割的 mask, hxw, np.array 形式.

#     h = output_probs.shape[0]
#     w = output_probs.shape[1]

#     output_probs = np.expand_dims(output_probs, 0)
#     output_probs = np.append(1 - output_probs, output_probs, axis=0)

#     d = dcrf.DenseCRF2D(w, h, 2)
#     U = -np.log(output_probs)
#     U = U.reshape((2, -1))
#     U = np.ascontiguousarray(U)
#     img = np.ascontiguousarray(img)

#     d.setUnaryEnergy(U)

#     d.addPairwiseGaussian(sxy=20, compat=3)
#     d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

#     Q = d.inference(5)
#     Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

#     return Q
# # """
# # 测试 demo
# # #image - 原始图片，hxwx3，采用 PIL.Image 读取
# # #seg_map - 假设为语义分割的 mask, hxw, np.array 形式.


# image = Image.open(r'G:\Remote_sensing_extract\H48F019017\Cut_0.tif')
# seg_map=Image.open(r'G:\Remote_sensing_extract\H48F019017\Unet\Cut_0.tif')

# final_mask = dense_crf_2d(np.array(image).astype(np.uint8), (np.array(seg_map).astype(np.uint8)))
# plt.subplot(1, 3, 1)
# plt.imshow(image)
# plt.subplot(1, 3, 2)
# plt.imshow(seg_map)
# plt.subplot(1, 3, 3)
# plt.imshow(final_mask)
# plt.show()
