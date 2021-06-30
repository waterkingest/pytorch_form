import cv2
import random
import os
import numpy as np
from tqdm import tqdm
 
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle,img_w,img_h):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb,img_w,img_h,num):
    # if np.random.random() < 0.25:
    if num==1:
        xb,yb = rotate(xb,yb,90,img_w,img_h)
    # if np.random.random() < 0.25:
    if num==2:
        xb,yb = rotate(xb,yb,180,img_w,img_h)
    # if np.random.random() < 0.25:
    if num==3:
        xb,yb = rotate(xb,yb,270,img_w,img_h)
    # if np.random.random() < 0.25:
    if num==4:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
        
    # if np.random.random() < 0.25:
    if num==5:
        xb = random_gamma_transform(xb,1.0)
        
    # if np.random.random() < 0.25:
    if num==6:
        xb = blur(xb)
    
    # if np.random.random() < 0.2:
    if num==7:
        xb = add_noise(xb)
        
    return xb,yb

def creat_dataset(image_num = 8, mode = 'original'):
    print('creating dataset...')
    imagepath=r'Datatest\img'
    labelpath=r'Datatest\label'
    temp_seg = os.listdir(labelpath)
    total_seg = []
    
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)
    g_count = len(total_seg)
    need_creat=image_num-len(total_seg)
    for i in tqdm(range(need_creat)):
        randomfile=np.random.randint(0,len(total_seg))
        src_img = cv2.imread(imagepath +'/'+ total_seg[randomfile])  # 3 channels
        label_img = cv2.imread(labelpath +'/'+ total_seg[randomfile],cv2.IMREAD_GRAYSCALE)  # single channel 
        img_h,img_w,_ = src_img.shape 

        if mode == 'augment':
            src_roi,label_roi = data_augment(src_img,label_img,img_w,img_h,g_count)
        cv2.imwrite(('Datatest\img/%d.png' % g_count),src_roi)
        cv2.imwrite(('Datatest\label/%d.png' % g_count),label_roi)
        g_count += 1

if __name__=='__main__':  
    creat_dataset(mode='augment')