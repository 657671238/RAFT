import os
from PIL import Image
import cv2
from torch.utils import data
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import warnings
import random
import numpy

def modcrop(image, modulo):
    h, w = image.shape[0], image.shape[1]
    h = h - h % modulo
    w = w - w % modulo
    return image[:h, :w]

def cv2_rotate(image, angle=15):
	height, width = image.shape[:2]    
	center = (width / 2, height / 2)   
	scale = 1                        
	M = cv2.getRotationMatrix2D(center, angle, scale)
	image_rotation = cv2.warpAffine(src=image, M=M, dsize=(width, height), borderValue=(0, 0, 0))
	return image_rotation



def make_augment(low_quality, high_quality):
	# 以 0.6 的概率作数据增强
	if(random.random() > 1 - 0.9):
		# 待增强操作列表(如果是 Unet 的话, 其实这里可以加入一些旋转操作)
		all_states = ['crop', 'flip', 'rotate']
		# 打乱增强的顺序
		random.shuffle(all_states)
		for cur_state in all_states:
			if(cur_state == 'flip'):
				# 0.5 概率水平翻转
				if(random.random() > 0.5):
					low_quality = cv2.flip(low_quality, 1)
					high_quality = cv2.flip(high_quality, 1)
					# print('水平翻转一次')
			elif(cur_state == 'crop'):
				# 0.5 概率做裁剪
				if(random.random() > 1 - 0.8):
					H, W, _ = low_quality.shape
					ratio = random.uniform(0.75, 0.95)
					_H = int(H * ratio)
					_W = int(W * ratio)
					pos = (numpy.random.randint(0, H - _H), numpy.random.randint(0, W - _W))
					low_quality = low_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					high_quality = high_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					# print('裁剪一次')
			elif(cur_state == 'rotate'):
				# 0.2 概率旋转
				if(random.random() > 1 - 0.1):
					angle = random.randint(-15, 15)  
					low_quality = cv2_rotate(low_quality, angle)
					high_quality = cv2_rotate(high_quality, angle)
					# print('旋转一次')
	return low_quality, high_quality

def cut_img(low,high):
    h, w = low.shape[0], low.shape[1]
    l = random.randint(0,h-224)
    r = random.randint(0,w-224)
    low_cut = low[l:l+224,r:r+224,:]
    high_cut = high[l:l+224,r:r+224,:]
    return low_cut,high_cut


class datset(Dataset):
    """NYUDataset."""

    def __init__(self, train=True, size=14):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            数据必须被8整除
        """
        self.size = size
        self.start = 0
        self.root_dir = '/home/dx/usrs/dongkun/dataset/noise/sintel'
        self.nirs = self.root_dir + '/color_0'
        self.uvs = self.root_dir + '/warp_uv'
        self.occus = self.root_dir + '/occultation'
        #self.noises = self.root_dir + '/3dLUT'
        # if train == False:
        #     self.noises = self.root_dir + '/3dLUT'
        self.transform = transforms.Compose([
           transforms.ToTensor()
        ])
        list1 = os.listdir(self.nirs)
        list2 = os.listdir(self.uvs)
        list3 = os.listdir(self.occus)
        self.start = 0
        self.lens = 900
        if train == False:
            self.start = 900
            self.lens = len(list1)-900
        # self.start = 0
        # self.lens = 5
        list1.sort(key=lambda x: int(x[:-4]))
        list2.sort(key=lambda x: int(x[:-4]))
        list3.sort(key=lambda x: int(x[:-4]))
        self.list1 = [os.path.join(self.nirs, img) for img in list1]
        self.list2 = [os.path.join(self.uvs, img) for img in list2]
        self.list3 = [os.path.join(self.occus, img) for img in list3]


        self.gts = '/home/dx/usrs/dongkun/dataset/GT/level0/sintel/color_0_rgb'
        list4 = os.listdir(self.gts)
        list4.sort(key=lambda x: int(x[:-4]))
        self.list4 = [os.path.join(self.gts, img) for img in list4]

        self.rig = '/home/dx/usrs/dongkun/dataset/noise/sintel/color_1'
        list5 = os.listdir(self.rig)
        list5.sort(key=lambda x: int(x[:-4]))
        self.list5 = [os.path.join(self.rig, img) for img in list5]
        random.seed(2)
        random.shuffle(self.list1)
        random.seed(2)
        random.shuffle(self.list2)
        random.seed(2)
        random.shuffle(self.list3)
        random.seed(2)
        random.shuffle(self.list4)
        random.seed(2)
        random.shuffle(self.list5)
        # print(self.list1[:10])
        # print(self.list2[:10])
        # print(self.list3[:10])
        # print(self.list4[:10])

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
        nir = self.list1[self.start+idx]
        uv = self.list2[self.start+idx]
        occ = self.list3[self.start+idx]
        gt = self.list4[self.start+idx]
        right = self.list5[self.start+idx]



        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nir = cv2.imread(nir,cv2.IMREAD_GRAYSCALE)
            # nir = Image.open(imfile)
            # print(nir.shape)
            # print(nir[:,:,0]==nir[:,:,1])
            uv = np.load(uv)
            occ = cv2.imread(occ,cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt)
            gt = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_RGB2YCR_CB)
            right = cv2.imread(right)
            # cv2.imwrite('./test_t.png',right.astype('uint8'))
            right = cv2.cvtColor(right.astype(np.uint8), cv2.COLOR_RGB2YCR_CB)
            # print(gt.dtype)
            # exit()
        # cv2.imwrite('./test.png',cv2.cvtColor(right.astype('uint8'),cv2.COLOR_YCrCb2RGB))
        nir = np.asarray(nir)
        occ = np.asarray(occ)
        uv = np.asarray(uv)
        right = np.asarray(right)
        #print(image.shape,noise.shape,type(image))
        # image = modcrop(image,28) 
        # noise = modcrop(noise,28) 
        # print(nir.shape)
        if self.transform:
            nir = self.transform(nir)
            uv = self.transform(uv)
            occ = self.transform(occ)
            gt = self.transform(gt)
            right = self.transform(right)
        #print(image.shape,noise.shape)

        # sample = {'gt': image, 'target': noise}

        return {'gt': gt, 'nir': nir, 'uv': uv, 'occ': occ,'right': right}


if __name__ == "__main__":
    test_dataset = datset()
    
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    for idx, i in enumerate(dataloader_test):
        gt = i['gt']
        targ = i['nir']
        uv = i['uv']
        occ = i['occ']
        print(gt.shape, targ.shape, uv.shape, occ.shape)
        if idx == 10:
            cv2.imwrite('1.png',gt[0].permute(1,2,0).cpu().numpy()*255.0)
            cv2.imwrite('2.png',uv[0,0].cpu().numpy()*255.0)
            cv2.imwrite('3.png',uv[0,1].cpu().numpy()*255.0)
            cv2.imwrite('4.png',occ[0,0].cpu().numpy()*255.0)
            # cv2.imwrite('5.png',gt[0].permute(1,2,0).cpu().numpy())
            
            exit()
