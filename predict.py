'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要原图和分割图不混合，可以把blend参数设置成False。
4、如果想根据mask获取对应的区域，可以参考detect_image中，利用预测结果绘图的部分。
seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
for c in range(self.num_classes):
    seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
    seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
    seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
'''
from PIL import Image
import time
import cv2
from unet import Unet
# from deeplab_predict import DeepLab 
import os
import numpy as np
unet = Unet()
# deeplab=DeepLab()
def List_predict():
    imagepath='F:\\exciseData\\testimage\\split'
    savepath='F:\\exciseData\\result\\Unet\\'
    imagelist=os.listdir(imagepath)
    total=len(imagelist)
    num=1
    print('Star!')
    for img in imagelist:
        if img.endswith('.png'):
            path1=savepath+img.split('.')[0]+'_result.png'
            imgpath=os.path.join(imagepath,img)
            image = Image.open(imgpath)
            # print(img)
            r_image = unet.detect_image(image)
            # r_image = deeplab.detect_image(image)
            r_image.save(path1)
        print(img,' 已完成 {:.2f}% '.format(num/total*100))
        num+=1
    print('finish!')
def single_predict():
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
            # print(type(image))
            # print(image.size)
            im1=np.array(image)
            # print(im1.shape)
            # print(type(im1))
            # old_image=Image.open(img.replace('JPEGImages','SegmentationClass'))
            # old_image.show()
        except:         
            print('Open Error! Try again!')
            continue
        else:
            r_image = unet.detect_image(image)
            # r_image=deeplab.detect_image(image)
            r_image.show()
def merge_predict():
    imagepath=r'F:\exciseData\testfile'
    savepath=r'F:\exciseData\testfile\result\\'
    num=8#水平分为几个
    imagelist=os.listdir(imagepath)
    total=len(imagelist)
    name=0
    print('Star!')
    for i in imagelist:
        # if i =='Cut_28.tif':
        if i.endswith('.tif') :
            filepath = os.path.join(imagepath, i)
            filename = i.split('.')[0].replace('label','')
            savepath2=savepath+filename+".png"           
            img = cv2.imread(filepath)
            [h, w] = img.shape[:2]
            print(filepath, (h, w))
            img_result = np.zeros([h, w], dtype=np.uint8)
            i=0          
            for i in range(1,num+1):
                for j in range(1,num+1):                  
                    # path1=os.path.join(newdir, filename) +"_"+str(name) +"_sat.png"
                    l1img = img[int((i-1)*h / num):int(i*h / num+1), int((j-1)*w / num):int(j*w / num+1),:]
                    
                    l2img=Image.fromarray(np.uint8(l1img))
                    r_image=deeplab.detect_image(l2img)
                    img_result[int((i-1)*h / num):int(i*h / num+1), int((j-1)*w / num):int(j*w /num+1)]=r_image
                    # l1img=cv2.cvtColor(l1img, cv2.COLOR_BGR2GRAY)
                    # print(path1,l1img.shape)                   
            name+=1
            img_result=Image.fromarray(np.uint8(img_result))
            # img_result.show()
            img_result.save(savepath2)
            print(savepath2,' 已完成 {:.2f}% '.format(name/150*100))
            # break
    print('finish!')
if __name__ == "__main__":
    # merge_predict()
    single_predict()