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
    warp_uv_dir = '/home/dx/usrs/dongkun/dataset/noise/sintel/warp_uv/'
    nir_dir = '/home/dx/usrs/dongkun/dataset/noise/sintel/color_0/'
    occ_dir = '/home/dx/usrs/dongkun/dataset/noise/sintel/occultation/'
    l2r_dir = '/home/dx/usrs/dongkun/dataset/noise/sintel/flow_l2r/'
    r2l_dir = '/home/dx/usrs/dongkun/dataset/noise/sintel/flow_r2l/'

    nir = os.listdir(nir_dir)
    warp_uv = os.listdir(warp_uv_dir)
    occ = os.listdir(occ_dir)
    l2r = os.listdir(l2r_dir)
    r2l = os.listdir(r2l_dir)

    nir = sorted(nir)
    warp_uv = sorted(warp_uv)
    l2r = sorted(l2r)
    r2l = sorted(r2l)
    occ = sorted(occ)
    for ii in range(len(nir)):
        n = cv2.imread(nir_dir+nir[ii],cv2.IMREAD_GRAYSCALE)
        n = n[:,:,None].astype(np.uint8)
        w = np.load(warp_uv_dir+warp_uv[ii])
        o = cv2.imread(occ_dir+occ[ii],cv2.IMREAD_GRAYSCALE)//255
        print(np.max(o), np.min(o))
        # print(o.dtype, w.dtype)
        # print((o//255).dtype)
        # exit()
        print(n.dtype, w.shape,o.shape)
        l = np.load(l2r_dir+l2r[ii])
        r = np.load(r2l_dir+r2l[ii])
        w = (w*o[:,:,None]).astype(np.uint8)
        print(n.dtype,w.dtype)
        pic = np.concatenate((n,w), 2)
        print(np.max(pic), np.min(pic))
        pic = cv2.cvtColor(pic.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        print(np.max(pic), np.min(pic))
        np.clip(pic, 0, 255)
        print(np.max(pic), np.min(pic))
        # cv2.imwrite('crop_.png', pic)
        # print(n.shape, w.shape, l.shape, r.shape)
        # print(np.max(l[:,:,:,0]), np.min(l[:,:,:,0]))
        # print(np.max(l[:,:,:,1]), np.min(l[:,:,:,1]))
        jl = np.sqrt(l[:,:,:,0]**2+r[:,:,:,1]**2)
        jl = jl[0]

        crop_num = 5
        min_ls = np.min(jl)
        max_ls = np.max(jl)
        
        left_o = [[0 for i in range(len(o[0]))] for j in range(len(o))]
        
        for i in range(len(o)):
            pre_jl = max_ls
            for j in range(len(o[0])):
                if o[i][j]==0:
                    left_o[i][j]=pre_jl
                else:
                    pre_jl = jl[i][j]

        right_o = [[0 for i in range(len(o[0]))] for j in range(len(o))]
        for i in range(len(o)):
            pre_jl = max_ls
            for j in range(len(o[0])-1, -1, -1):
                if o[i][j]==0:
                    right_o[i][j] = min(pre_jl, left_o[i][j])
                else:
                    pre_jl = jl[i][j]
        print(np.max(jl), np.min(jl))
        print(np.max(o), np.min(o))
        print(np.max(right_o), np.min(right_o))
        jl= jl*o+right_o

        print(jl.shape, o.shape)
        # exit()
        # exit()
        # print(l[0,1,:10,0])
        # print(l[0,1,:10,1])
        # print(jl[0,1,:10])
        print(np.max(jl), np.min(jl))



        length = (max_ls - min_ls)/5
        print(pic.shape, jl.shape, o.shape)
        # pic = cv2.cvtColor(pic.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        # cv2.imwrite('crop_.png', pic)
        ans = [pic*np.where((jl>(min_ls+i*length)) & (jl<=(min_ls+(i+1)*length)), 1, 0)[:,:,None] for i in range(crop_num)]
        ans = [np.where((jl>(min_ls+i*length)) & (jl<=(min_ls+(i+1)*length)), 1, 0)[:,:,None]*255 for i in range(crop_num)]


        print(len(ans))
        print(ans[0].shape)
        # print(ts.index(ts==2))
        # cv2.imwrite('./crop_dir/crop_all_{}.png'.format(ii), np.concatenate((pic,np.concatenate(ans, 0)),0), [cv2.IMWRITE_PNG_COMPRESSION, 0])

        cv2.imwrite('./crop_mask/{}_0.png'.format(str(ii).zfill(5)), ans[0])
        cv2.imwrite('./crop_mask/{}_1.png'.format(str(ii).zfill(5)), ans[1])
        cv2.imwrite('./crop_mask/{}_2.png'.format(str(ii).zfill(5)), ans[2])
        cv2.imwrite('./crop_mask/{}_3.png'.format(str(ii).zfill(5)), ans[3])
        cv2.imwrite('./crop_mask/{}_4.png'.format(str(ii).zfill(5)), ans[4])







