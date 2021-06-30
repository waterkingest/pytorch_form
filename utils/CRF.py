#coding:utf-8
import numpy as np
import pydensecrf.densecrf as dcrf

def dense_crf(img, output_probs): #img为输入的图像，output_probs是经过网络预测后得到的结果
    h = output_probs.shape[0] #高度
    w = output_probs.shape[1] #宽度

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2) #NLABELS=2两类标注，车和不是车
    U = -np.log(output_probs) #得到一元势
    U = U.reshape((2, -1)) #NLABELS=2两类标注
    U = np.ascontiguousarray(U) #返回一个地址连续的数组
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U) #设置一元势

    d.addPairwiseGaussian(sxy=20, compat=3) #设置二元势中高斯情况的值
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)#设置二元势众双边情况的值

    Q = d.inference(5) #迭代5次推理
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w)) #得列中最大值的索引结果

    return Q