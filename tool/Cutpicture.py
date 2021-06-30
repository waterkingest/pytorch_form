import cv2
import os
import numpy as np

if __name__ == "__main__":
    num=8#水平分为几个
    width=1024
    #图像文件原始路径
    path = r"F:\exciseData\water-20\img"  
    listdir = os.listdir(path)
    # 新建split文件夹用于保存
    newdir = os.path.join(path, 'split500')
    if (os.path.exists(newdir) == False):
        os.mkdir(newdir)
    for i in listdir:
        if i.endswith('.tif') :
            filepath = os.path.join(path, i)
            filename = i.split('.')[0].replace('label','')
            name=1
            img = cv2.imread(filepath)
            [h, w] = img.shape[:2]
            print(filepath, (h, w))
            i=0
           
            for i in range(1,num+1):
                for j in range(1,num+1):
                    # path1=os.path.join(newdir, filename) +"_"+str(name) +"_sat.png"
                    path1=os.path.join(newdir, filename) +"_"+str(name) +".png"
                    l1img = img[int((i-1)*h / num):int(i*h / num+1), int((j-1)*w / num):int(j*w / num+1), :]
                    # l1img=cv2.cvtColor(l1img, cv2.COLOR_BGR2GRAY)
                    # print(path1,l1img.shape)
                    cv2.imwrite(path1, l1img)
                    name+=1
print('finish!')