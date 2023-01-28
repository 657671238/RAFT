from __future__ import print_function
import os
import time
import socket
import pandas as pd
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from unet import UNet
# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from dataloader import datset
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate. default=0.0001')
parser.add_argument('--data_augmentation', type=bool, default=False, help='if adopt augmentation when training')
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR', help='the training dataset')
parser.add_argument('--Ispretrained', type=bool, default=False, help='If load checkpoint model')
parser.add_argument('--pretrained_sr', default='noise25.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', default='./checkpoint/', help='Location to load checkpoint models')
parser.add_argument("--noiseL", type=float, default=50, help='noise level')
parser.add_argument('--save_folder_u', default='./checkpoint_u/', help='Location to save checkpoint models')
parser.add_argument('--save_folder_v', default='./checkpoint_v/', help='Location to save checkpoint models')
parser.add_argument('--statistics_u', default='./statistics_u/', help='Location to save statistics')
parser.add_argument('--statistics_v', default='./statistics_v/', help='Location to save statistics')

# Testing settings
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size, default=1')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--test_dataset', type=str, default='Set12', help='the testing dataset')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')

# Global settings
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
parser.add_argument('--data_dir', type=str, default='/home/dongxuan/users/xukang/SIDD_Medium_Srgb', help='the dataset dir')
parser.add_argument('--model_type', type=str, default='CURTransformer', help='the name of model')
parser.add_argument('--Isreal', default=True, help='If training/testing on RGB images')
parser.add_argument('--csvfile', type=str, default='hello.csv', help='csv_files')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

if not os.path.exists(opt.save_folder_u):
    os.makedirs(opt.save_folder_u)
if not os.path.exists(opt.save_folder_v):
    os.makedirs(opt.save_folder_v)
if not os.path.exists(opt.statistics_u):
    os.makedirs(opt.statistics_u)
if not os.path.exists(opt.statistics_v):
    os.makedirs(opt.statistics_v)

def train_u(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        gt = batch['gt'].cuda()
        nir = batch['nir'].cuda()
        uv = batch['uv'].cuda()
        occ = batch['occ'].cuda()
        uv[:,0] = uv[:,0]*occ[:,0]+(occ[:,0]-1)
        input = torch.cat([nir, uv[:,:1]], 1)
        model.zero_grad()
        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)
        prediction = prediction * (1-occ)
        gt = gt[:,1:2] * (1-occ)
        loss = criterion(prediction, gt)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def train_v(epoch):
    epoch_loss = 0
    model2.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        gt = batch['gt'].cuda()
        nir = batch['nir'].cuda()
        uv = batch['uv'].cuda()
        occ = batch['occ'].cuda()
        uv[:,1] = uv[:,1]*occ[:,0]+(occ[:,0]-1)
        input = torch.cat([nir, uv[:,1:]], 1)
        model2.zero_grad()
        optimizer2.zero_grad()
        t0 = time.time()
        prediction = model2(input)
        prediction = prediction * (1-occ)
        gt = gt[:,2:3] * (1-occ)
        loss = criterion(prediction, gt)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer2.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))



def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def test_u(testing_data_loader):
    psnr_test= 0
    model.eval()
    for batch in testing_data_loader:
        gt = batch['gt'].cuda()
        nir = batch['nir'].cuda()
        uv = batch['uv'].cuda()
        occ = batch['occ'].cuda()
        with torch.no_grad():
            uv[:,0] = uv[:,0]*occ[:,0]+(occ[:,0]-1)
            input = torch.cat([nir, uv[:,:1]], 1)
            prediction = model(input)
            prediction = prediction * (1-occ)
            gt = gt[:,1:2] * (1-occ)
        # loss = criterion(prediction, gt)
        psnr_test += batch_PSNR(prediction, gt, 1.)
    print("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(testing_data_loader)))
    return psnr_test / len(testing_data_loader)

def test_v(testing_data_loader):
    psnr_test= 0
    model2.eval()
    for batch in testing_data_loader:
        gt = batch['gt'].cuda()
        nir = batch['nir'].cuda()
        uv = batch['uv'].cuda()
        occ = batch['occ'].cuda()
        with torch.no_grad():
            uv[:,1] = uv[:,1]*occ[:,0]+(occ[:,0]-1)
            input = torch.cat([nir, uv[:,1:2]], 1)
            prediction = model2(input)
            prediction = prediction * (1-occ)
            gt = gt[:,2:3] * (1-occ)
        psnr_test += batch_PSNR(prediction, gt, 1.)
    print("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(testing_data_loader)))
    return psnr_test / len(testing_data_loader)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)




if __name__ == '__main__':
    print('===> Loading datasets')


    train_set = datset(train=True)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=4,drop_last=True)
    
    # exit()
    test_set = datset(train=False)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False, num_workers=0, drop_last=True)

    print('===> Building model ', opt.model_type)
    model = UNet(3,1)
    model2 = UNet(3,1)

    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    model2 = torch.nn.DataParallel(model2, device_ids=gpus_list)
    criterion = nn.L1Loss()

    print('---------- Networks architecture -------------')
    print_network(model)
    print('----------------------------------------------')

    model_name_u = '/home/dx/usrs/dongkun/xk/colour/lr_check_wx/checkpoint_u/ep200.pth'
    model_name_v = '/home/dx/usrs/dongkun/xk/colour/lr_check_wx/checkpoint_v/ep200.pth'
    model.load_state_dict(torch.load(model_name_u, map_location=lambda storage, loc: storage))
    model2.load_state_dict(torch.load(model_name_v, map_location=lambda storage, loc: storage))
        # print(model_name + ' model is loaded.')

    min_psnr = -1
    PSNR = []
    epoch_loss = 0
    model2.eval()
    model.eval()
    psnr_test= 0
    for idx, batch in enumerate(testing_data_loader):
        gt = batch['gt'].cuda()
        nir = batch['nir'].cuda()
        uv = batch['uv'].cuda()
        occ = batch['occ'].cuda()
        right = batch['right'].cuda()
        with torch.no_grad():
            uv[:,0] = uv[:,0]*occ[:,0]+(occ[:,0]-1)
            input = torch.cat([nir, uv[:,:1], occ], 1)
            prediction = model(input)
            # prediction = prediction * (1-occ)
            # u_new = uv[:,:1]*occ+prediction
            u_new = prediction


            uv[:,1] = uv[:,1]*occ[:,0]+(occ[:,0]-1)
            input = torch.cat([nir, uv[:,1:2], occ], 1)
            prediction = model2(input)
            # prediction = prediction * (1-occ)
            # gt = gt[:,2:3] * (1-occ)
            # v_new = uv[:,1:2]*occ+prediction
            v_new = prediction

            out = torch.cat([nir,u_new,v_new],1)
            out_pre = torch.cat([nir,uv],1)
            # cv2.imwrite('./test/ans_nir.png',nir[0][0].cpu().numpy()*255)
            # cv2.imwrite('./test/ans_occ.png',occ[0][0].cpu().numpy()*255)
            # cv2.imwrite('./test/ans.png',cv2.cvtColor((out[0].permute(1, 2, 0).cpu().numpy()*255).astype('uint8'),cv2.COLOR_YCrCb2RGB))
            cv2.imwrite('./test/pre'+str(idx).zfill(5)+'.png',cv2.cvtColor((torch.cat([right[0] ,out_pre[0], out[0], gt[0]],1).permute(1, 2, 0).cpu().numpy()*255).astype('uint8'),cv2.COLOR_YCrCb2RGB))
            cv2.imwrite('./test/pre_u'+str(idx).zfill(5)+'.png',(torch.cat([out_pre[0], out[0], gt[0]],1).permute(1, 2, 0)[:,:,1].cpu().numpy()*255).astype('uint8'))
            cv2.imwrite('./test/pre_v'+str(idx).zfill(5)+'.png',(torch.cat([out_pre[0], out[0], gt[0]],1).permute(1, 2, 0)[:,:,2].cpu().numpy()*255).astype('uint8'))

        # exit()
        # psnr_test += batch_PSNR(prediction, gt, 1.)
    # print("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(testing_data_loader)))


    # for epoch in range(opt.start_iter, opt.nEpochs + 1):
    #     train_u(epoch)
    #     psnr = test_u(testing_data_loader)
    #     scheduler.step()
    #     PSNR.append(psnr)
    #     data_frame = pd.DataFrame(
    #         data={'epoch': epoch, 'PSNR': PSNR}, index=range(opt.start_iter, epoch+1)
    #     )
    #     data_frame.to_csv(os.path.join(opt.statistics_u, opt.model_type + opt.csvfile), index_label='index')
    #     # learning rate is decayed by a factor of 10 every half of total epochs
    #     if (epoch + 1) % (opt.nEpochs / 2) == 0:
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] /= 10.0
    #         print('Learning rate decay: lr={}'.format(param_group['lr']))
    #     if min_psnr==-1:
    #         min_psnr = psnr
    #     else:
    #         if psnr > min_psnr:
    #             images_files = os.listdir(opt.save_folder_u)
    #             for file in images_files:
    #                 if file.endswith('.pth'):
    #                     os.remove(os.path.join(opt.save_folder_u, file))
    #             min_psnr = psnr
    #             model.eval()
    #             SC = opt.model_type + 'net_epoch_' + str(epoch) + '_'  + '.pth'
    #             torch.save(model.state_dict(), os.path.join(opt.save_folder_u, SC))
    #             model.train()
    #         lastest_model = 'ep' + str(epoch) + '.pth'
    #         torch.save(model.state_dict(), os.path.join(opt.save_folder_u, SC))




    # optimizer2 = optim.Adam(model2.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.2)
    # min_psnr = -1
    # PSNR = []
    # for epoch in range(opt.start_iter, opt.nEpochs + 1):
    #     train_v(epoch)
    #     psnr = test_v(testing_data_loader)
    #     scheduler2.step()
    #     PSNR.append(psnr)
    #     data_frame = pd.DataFrame(
    #         data={'epoch': epoch, 'PSNR': PSNR}, index=range(opt.start_iter, epoch+1)
    #     )
    #     data_frame.to_csv(os.path.join(opt.statistics_v, opt.model_type + opt.csvfile), index_label='index')
    #     # learning rate is decayed by a factor of 10 every half of total epochs
    #     if (epoch + 1) % (opt.nEpochs / 2) == 0:
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] /= 10.0
    #         print('Learning rate decay: lr={}'.format(param_group['lr']))
    #     if min_psnr==-1:
    #         min_psnr = psnr
    #     else:
    #         if psnr > min_psnr:
    #             images_files = os.listdir(opt.save_folder_v)
    #             for file in images_files:
    #                 if file.endswith('.pth'):
    #                     os.remove(os.path.join(opt.save_folder_v, file))
    #             min_psnr = psnr
    #             model.eval()
    #             SC = opt.model_type + 'net_epoch_' + str(epoch) + '_'  + '.pth'
    #             torch.save(model.state_dict(), os.path.join(opt.save_folder_v, SC))
    #             model.train()
    #         lastest_model = 'ep' + str(epoch) + '.pth'
    #         torch.save(model2.state_dict(), os.path.join(opt.save_folder_v, SC))