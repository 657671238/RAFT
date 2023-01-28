import shutil 
from ast import Num
from re import L
import re
import time
import sys
sys.path.append('core')
import os
from warp import image_warp
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from PIL import Image
import torch.nn.functional as F
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import cv2

DEVICE = 'cuda'
parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()

def load_image(imfile):
    # img = np.array(Image.open(imfile).convert('L')).astype(np.uint8)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = np.array(Image.open(imfile)).astype(np.uint8)
    
    img = img/255
    img = (img[:,:,::-1]**(2.2))*30
    img = img**(1/2.2)
    img = img*255
    img = np.clip(img, 0, 255)
    
    print(img.shape)
    cv2.imwrite('./right_10.png',img)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YCR_CB)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    img1 = img[0].unsqueeze(0).expand(3,-1,-1)
    # img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = torch.from_numpy(img).permute(2,0,1).float()
    return img1[None].to(DEVICE), img[None].to(DEVICE)

def load_image_gray(imfile):
    # img = np.array(Image.open(imfile).convert('L')).astype(np.uint8)
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # print(img.shape)
    img = torch.from_numpy(img).float().unsqueeze(0).expand(3,-1,-1)

    return img[None].to(DEVICE)

def viz(img, flo):
    result_0 = image_warp(img.permute(0,2,3,1).cpu().numpy(),flo.permute(0,2,3,1).cpu().numpy())
    return result_0

def feature_warp_ori(imageL,imageR,featureL,use_model='things'):
    '''
    特征对其函数
    输入L和R两张图像和图像L的特征，将L的特征向R对齐
    model=things or small
    如果显存不够可以使用small，效果和things差距不大
    '''
    if use_model=='small':
        args.small=True
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('models/raft-{}.pth'.format(use_model)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        padder = InputPadder(imageL.shape)
        imageL, tmp = padder.pad(imageL, imageR)
        imageR,featureL=padder.pad(imageR,featureL)
        flow_low, flow_up = model(imageR, imageL, iters=20, test_mode=True)
        result=viz(featureL, flow_up)
        return padder.unpad(result)

def feature_warp(imageL,imageR,featureL,use_model='things'):
    '''
    特征对其函数
    输入L和R两张图像和图像L的特征，将L的特征向R对齐
    model=things or small
    如果显存不够可以使用small，效果和things差距不大
    '''
    imgl_pre = featureL

    if use_model=='small':
        args.small=True
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('models/raft-{}.pth'.format(use_model)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    featureL = imageL
    imageL_ori = imageL
    imageR_ori = imageR
    with torch.no_grad():
        padder = InputPadder(imageL.shape)
        imageL, imgl_pre = padder.pad(imageL, imgl_pre)
        imageR,featureL=padder.pad(imageR,featureL)
        flow_low, flow_l2r = model(imageR, imageL, iters=20, test_mode=True)
        result_l2r=viz(imgl_pre, flow_l2r)
        result_ori = result_l2r
        result_l2r = torch.from_numpy(result_l2r).float().permute(0, 3, 1, 2).to(DEVICE)
    featureR = result_l2r
    imageL = imageL_ori
    with torch.no_grad():
        padder = InputPadder(result_l2r.shape)
        result_l2r, tmp = padder.pad(result_l2r, imageL)
        imageL,featureR=padder.pad(imageL,featureR)
        flow_low, flow_up_12r2l = model(imageL, result_l2r, iters=20, test_mode=True)

    return padder.unpad(result_ori), flow_l2r, flow_up_12r2l

def feature_warp_cycle_ori(imageL,imageR,use_model='things'):
    '''
    特征对其函数
    输入L和R两张图像，将L的特征向R对齐,然后再将L向R对齐的结果再向L对齐，以此判定光流错误区域
    model=things or small
    如果显存不够可以使用small，效果和things差距不大
    '''
    if use_model=='small':
        args.small=True
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('models/raft-{}.pth'.format(use_model)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    #L2R & R2L
    featureL = imageL
    imageL_ori = imageL
    imageR_ori = imageR
    with torch.no_grad():
        padder = InputPadder(imageL.shape)
        imageL, tmp = padder.pad(imageL, imageR)
        imageR,featureL=padder.pad(imageR,featureL)
        flow_low, flow_up = model(imageR, imageL, iters=20, test_mode=True)
        result_l2r=viz(featureL, flow_up)
        result_l2r_ori = result_l2r
        result_l2r = torch.from_numpy(result_l2r).float().permute(0, 3, 1, 2).to(DEVICE)

    featureR = imageR
    imageL = imageL_ori
    imageR = imageR_ori
    with torch.no_grad():
        padder = InputPadder(imageR.shape)
        imageR, tmp = padder.pad(imageR, imageL)
        imageL,featureR=padder.pad(imageL,featureR)
        tmp,result_l2r=padder.pad(imageL,result_l2r)
        flow_low, flow_up = model(imageL, imageR, iters=20, test_mode=True)
        result_l2r2l=viz(result_l2r, flow_up)
    return padder.unpad(result_l2r_ori),padder.unpad(result_l2r2l)

def feature_warp_cycle_12_14(imageL, imageR, use_model='things'):
    '''
    特征对其函数
    输入L和R两张图像，将L的特征向R对齐,然后再将L向R对齐的结果再向L对齐，以此判定光流错误区域
    model=things or small
    如果显存不够可以使用small，效果和things差距不大
    '''
    if use_model=='small':
        args.small=True
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('models/raft-{}.pth'.format(use_model)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    #L2R & R2L
    featureL = imageL
    imageL_ori = imageL
    imageR_ori = imageR
    with torch.no_grad():
        padder = InputPadder(imageL.shape)
        imageL, tmp = padder.pad(imageL, imageR)
        imageR,featureL=padder.pad(imageR,featureL)
        flow_low, flow_up_l2r = model(imageR, imageL, iters=20, test_mode=True)
        result_l2r=viz(featureL, flow_up_l2r)
        result_l2r_ori = result_l2r
        result_l2r = torch.from_numpy(result_l2r).float().permute(0, 3, 1, 2).to(DEVICE)
    
    featureR = result_l2r
    imageL = imageL_ori
    with torch.no_grad():
        padder = InputPadder(result_l2r.shape)
        result_l2r, tmp = padder.pad(result_l2r, imageL)
        imageL,featureR=padder.pad(imageL,featureR)
        flow_low, flow_up_l2r2l = model(imageL, result_l2r, iters=20, test_mode=True)
        result_l2r2l=viz(featureR, flow_up_l2r2l)
    return flow_up_l2r, flow_up_l2r2l, padder.unpad(result_l2r_ori),padder.unpad(result_l2r2l)

def feature_warp_cycle(imageL, imageR, use_model='things'):
    '''
    特征对其函数
    输入L和R两张图像，将L的特征向R对齐,然后再将L向R对齐的结果再向L对齐，以此判定光流错误区域
    model=things or small
    如果显存不够可以使用small，效果和things差距不大
    '''
    if use_model=='small':
        args.small=True
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('models/raft-{}.pth'.format(use_model)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    #L2R & R2L
    featureL = imageL
    imageL_ori = imageL
    imageR_ori = imageR
    with torch.no_grad():
        padder = InputPadder(imageL.shape)
        imageL, tmp = padder.pad(imageL, imageR)
        imageR,featureL=padder.pad(imageR,featureL)
        flow_low, flow_up_l2r = model(imageR, imageL, iters=20, test_mode=True)
        result_l2r=viz(featureL, flow_up_l2r)
    
    featureR = imageR_ori
    imageL = imageL_ori
    imageR = imageR_ori
    with torch.no_grad():
        padder = InputPadder(imageR.shape)
        imageR, tmp = padder.pad(imageR, imageL)
        imageL,featureR=padder.pad(imageL,featureR)
        flow_low, flow_up_r2l = model(imageL, imageR, iters=20, test_mode=True)

    return flow_up_l2r, flow_up_r2l, padder.unpad(result_l2r)

def judge_difference(img1_list, img2_list):
    num, c, h, w = img1_list.shape
    mask_list = []
    for i in range(num):
        mask_temp = np.zeros((h, w), np.uint8)
        for x in range(h):
            for y in range(w):
                if (img1_list[i, 0, x, y] != img2_list[i, 0, x, y]) or (img1_list[i, 1, x, y] != img2_list[i, 1, x, y]) or (img1_list[i, 2, x, y] != img2_list[i, 2, x, y]):
                    mask_temp[x, y] = 255
        mask_list.append(mask_temp)
    return mask_list

def generate_locationmap(h, w):
    x=np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            x[i, j] = (i, j)
    x = torch.from_numpy(x).float()
    return x[None].to(DEVICE)

def compute_flow_occultation(flow1, flow2, threshold):
    num, h, w, c = flow1.shape
    mask = np.zeros((h, w), np.uint8)
    for i in range(0, h):
        for j in range(0, w):
            #读取该点光流信息
            flow_x = float(j + flow1[0, i, j, 0])
            flow_y = float(i + flow1[0, i, j, 1])
            if flow_x < (-0.0*(threshold)) or flow_y < (-0.0*(threshold)) or flow_x > ((w-1+0.0*threshold)*1.0) or flow_y > ((h-1+0.0*threshold)*1.0):
                mask[i, j] = 255
                continue
            #截断
            # print('截断前光流：x: ', flow_x, 'y: ', flow_y)
            flow_x = max(min(flow_x, (w-1)*1.0), 0)
            flow_y = max(min(flow_y, (h-1)*1.0), 0)
            # print('截断后光流：x: ', flow_x, 'y: ', flow_y)
            #上下界限
            x_low= math.floor(flow_x)
            x_high = min(x_low+1, w-1)
            y_low= math.floor(flow_y)
            y_high = min(y_low+1, h-1)
            # print('上下界限：x_low: ', x_low, 'x_high: ', x_high, 'y_low: ', y_low, 'y_high: ', y_high)
            #计算加权系数
            x_high_weight = flow_x - x_low
            x_low_weight = 1.0-x_high_weight
            y_high_weight = flow_y - y_low
            y_low_weight = 1.0-y_high_weight
            # print('加权系数：x_low_weight: ', x_low_weight, 'x_high_weight: ', x_high_weight, 'y_low_weight: ', y_low_weight, 'y_high_weight: ', y_high_weight)
            #求解反向光流
            flow_2_ll_x = flow2[0, y_low, x_low, 0]
            flow_2_lh_x = flow2[0, y_low, x_high, 0]
            flow_2_hl_x = flow2[0, y_high, x_low, 0]
            flow_2_hh_x = flow2[0, y_high, x_high, 0]
            # print('反向光流：flow_2_ll_x: ', flow_2_ll_x, 'flow_2_lh_x: ', flow_2_lh_x, 'flow_2_hl_x: ', flow_2_hl_x, 'flow_2_hh_x: ', flow_2_hh_x)
            x_low_value = x_low_weight*flow_2_ll_x + x_high_weight*flow_2_lh_x
            x_high_value = x_low_weight*flow_2_hl_x + x_high_weight*flow_2_hh_x
            x = y_low_weight*x_low_value+y_high_weight*x_high_value
            # print('x_low_value: ', x_low_value)
            # print('x_high_value: ', x_high_value)
            # print('flow_x: ', flow_x)
            # print('x: ', x)
            # print('j: ', j)
            # print('abs(flow_x+x-j): ', abs(flow_x+x-j))
            
            if (abs(flow_x+x-j)>threshold):
                mask[i, j] = 255
                continue
            
            flow_2_ll_y = flow2[0, y_low, x_low, 1]
            flow_2_lh_y = flow2[0, y_low, x_high, 1]
            flow_2_hl_y = flow2[0, y_high, x_low, 1]
            flow_2_hh_y = flow2[0, y_high, x_high, 1]
            # print('反向光流：flow_2_ll_y: ', flow_2_ll_y, 'flow_2_lh_y: ', flow_2_lh_y, 'flow_2_hl_y: ', flow_2_hl_y, 'flow_2_hh_y: ', flow_2_hh_y)
            
            y_low_value = x_low_weight*flow_2_ll_y + x_high_weight*flow_2_lh_y
            y_high_value = x_low_weight*flow_2_hl_y + x_high_weight*flow_2_hh_y
            y = y_low_weight*y_low_value+y_high_weight*y_high_value
            # print('y_low_value: ', y_low_value)
            # print('y_high_value: ', y_high_value)
            # print('y: ', y)
            # print('abs(flow_y+y-i): ', abs(flow_y+y-i))
            if (abs(flow_y+y-i)>threshold):
                mask[i, j] = 255
            # return
                
    return (255-mask)

if __name__ == '__main__':
    # img_path = 'images_12_10'
    # # 将imgl_path中的图像向imgr_path(主图像)对齐，
    result_save_path = 'results_test'
    # img_folder_names = os.listdir(img_path)
    # img_folder_names.sort()
    # img_num = 32
    # # print(img_folder_names)
    # for i in range(62, len(img_folder_names), 2):
    # try:
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    root1 = '/home/dx/usrs/dongkun/dataset/noise/sintel/color_0'
    root2 = '/home/dx/usrs/dongkun/dataset/noise/sintel/color_1'
    img1s = os.listdir(root1)
    img2s = os.listdir(root2)
    img1s = sorted(img1s)
    img2s = sorted(img2s)
    print(len(img1s),'_pictures')
    for i in range(766,len(img1s)):
        img1 = root1+'/'+img1s[i]
        img2 = root2+'/'+img2s[i]
        time1 = time.clock()
        # index = '%04d' % (i+1)
        # print(index)
        iml=load_image_gray(img1)
        imr, imr_pre=load_image(img2)
        # shutil.copyfile(img1,result_save_path+'/left.png')
        # shutil.copyfile(img2,result_save_path+'/right.png')
        print(iml[0].shape)

        # cv2.imwrite(result_save_path+'/right_nir.png',imr_pre[0][0].cpu().numpy())
        # exit()
        num, c, h, w = imr.shape

        time3 = time.clock()
        # result_l2r = feature_warp_ori(iml, imr, iml)
            
        # cv2.imwrite(result_save_path+'/imr_pre.png',cv2.cvtColor(imr_pre[0].permute(1, 2, 0).cpu().numpy(),cv2.COLOR_BGR2RGB))
        ori, flow_l2r, flow_up_12r2l = feature_warp(imr,iml,imr_pre)
        # print(ori.shape, flow_l2r.shape, flow_up_12r2l.shape)
        # cv2.imwrite(result_save_path+'/l2r.png',cv2.cvtColor(ori[0],cv2.COLOR_BGR2RGB))
        # cv2.imwrite(result_save_path+'/l2r0.png',ori[0][:,:,0])
        # cv2.imwrite(result_save_path+'/l2r1.png',ori[0][:,:,1])
        uv = ori[0][:,:,1:]
        
        np.save('/home/dx/usrs/dongkun/dataset/noise/sintel/warp_uv/'+img1s[i][:-4]+'.npy', uv)
        # ori, flow_l2r, flow_up_12r2l = feature_warp(imr,iml,imr)
        # print(ori.shape, flow_l2r.shape, flow_up_12r2l.shape)
        # cv2.imwrite(result_save_path+'/r2l.png',cv2.cvtColor(ori[0],cv2.COLOR_BGR2RGB))


        # exit()

        flow_l2r, flow_r2l, result_l2r =feature_warp_cycle(imr,iml)
        # flow_r2l, flow_r2l2r, result_r2l, result_r2l2r =feature_warp_cycle(imr,iml)
        flow_l2r = flow_l2r.permute(0, 2, 3, 1).cpu().numpy()
        flow_r2l = flow_r2l.permute(0, 2, 3, 1).cpu().numpy()
        np.save('/home/dx/usrs/dongkun/dataset/noise/sintel/flow_l2r/'+img1s[i][:-4]+'.npy', flow_l2r)
        np.save('/home/dx/usrs/dongkun/dataset/noise/sintel/flow_r2l/'+img1s[i][:-4]+'.npy', flow_r2l)
        threshold = 5.0
        occultation_l2r = compute_flow_occultation(flow_l2r, flow_r2l, threshold)
        time4 = time.clock()
        print('第%s张图片，光流运行时间:%s秒' % ((1), (time4 - time3)))
        
        cv2.imwrite(os.path.join('/home/dx/usrs/dongkun/dataset/noise/sintel/occultation', img1s[i][:-4]+'.png'),occultation_l2r, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # time2 = time.clock()
        # print('第%s张图片，程序运行时间:%s秒' % ((1), (time2 - time1)))
        # # print(flow_r2l)
        # # print("111111")
        # # print(flow_l2r) 
        # # exit()
        # occultation_l2r = occultation_l2r[:,:,None]//255
        # print(occultation_l2r.shape, uv.shape)
        # np.repeat(occultation_l2r, 3, 2)
        # # print(occultation_l2r[][])
        # uv = uv*occultation_l2r
        # iml = iml.permute(0, 2, 3, 1).cpu().numpy()
        # print(iml.dtype)
        # print(uv.dtype)
        
        # iml[0][:,:,1:] = uv
        # print(iml.dtype)
        # iml = iml.astype('uint8')
        # print(iml.dtype)
        # image = cv2.cvtColor(iml[0], cv2.COLOR_YCrCb2RGB)
        # cv2.imwrite(result_save_path+'/last.png',image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # exit()

        # # cv2.imwrite(result_save_path+'/last.png',cv2.cvtColor(cv2.cvtColor(ori[0],cv2.COLOR_BGR2RGB), cv2.COLOR_YCrCb2RGB))
        # cv2.imwrite(result_save_path+'/last.png',iml[0][:,:,::-1])


        # image = cv2.imread(result_save_path+'/last.png')  # 读入图片
        # image = image[:,:,::-1]

        # image = cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
        # cv2.imwrite(result_save_path+'/lastlast.png',image)

        # image = cv2.cvtColor(iml[0], cv2.COLOR_YCrCb2RGB)
        # cv2.imwrite(result_save_path+'/lastlast1.png',image)

        # img_num = 1 + 1


