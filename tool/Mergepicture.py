
import PIL.Image as Image
import os
import numpy as np
import cv2
if __name__ == "__main__":
    num=12#水平分为几个
    width=1024
    #图像文件原始路径
    path = r"F:\\exciseData\\result\\Deeplab"  #小图位置
    origan=r"F:\\exciseData\\testimage"#大图位置
    listdir = os.listdir(path)
    # 新建split文件夹用于保存
    mergelist=[]
    newdir = os.path.join(path, 'merge')
    if (os.path.exists(newdir) == False):
        os.mkdir(newdir)
    total=len(listdir)
    nm=0
    for i in listdir:
        if i.endswith('.png') :
            filepath = os.path.join(path, i)
            filename = i.split('.')[0].replace('_result','')
            realname='_'.join(filename.split('_')[0:-1])
            if realname not in mergelist:
                mergelist.append(realname)
                name=1
                print(origan+'//'+realname+'.tif')
                img = Image.open(origan+'//'+realname+'.tif')
                [w, h] = img.size
                img = np.zeros([h, w, 4], dtype=np.uint8)
                print((w, h))
                for i in range(1,num+1):
                    for j in range(1,num+1):
                        # path1=os.path.join(newdir, filename) +"_"+str(name) +"_sat.png"                      
                        # print(path+'\\'+realname+'_'+str(name)+'_result.png')
                        img[int((i-1)*h / num):int(i*h / num+1), int((j-1)*w / num):int(j*w /num+1), :]=Image.open(path+'\\'+realname+'_'+str(name)+'_result.png')
                        # l1img=cv2.cvtColor(l1img, cv2.COLOR_BGR2GRAY)  
                        name+=1
                path1=os.path.join(newdir, realname) +".png"
                im=Image.fromarray(img)
                im.save(path1)
                nm+=name
                print(path1,' 已完成 {:.2f}% '.format(nm/total*100))
            else:
                continue
            # break
print('finish!')