import math
import numpy as np
import logging
import cv2
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import imgvision as iv


class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        ### create a logger
        self.__logger = logging.getLogger(logger_name)

        ### set the log level
        self.__logger.setLevel(log_level)

        ### create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        ### create a handler to print on console
        console_handler = logging.StreamHandler()

        ### define the output format of handlers
        formatter = logging.Formatter('[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        ### add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def mkExpDir(args):
    if (os.path.exists(args.save_dir)):
        if (not args.reset):
            raise SystemExit('Error: save_dir "' + args.save_dir + '" already exists! Please set --reset True to delete the folder.')
        else:
            shutil.rmtree(args.save_dir)

    os.makedirs(args.save_dir)
    # os.makedirs(os.path.join(args.save_dir, 'img'))

    if ((not args.eval) and (not args.test)):
        os.makedirs(os.path.join(args.save_dir, 'model'))
    
    if ((args.eval and args.eval_save_results) or args.test):
        os.makedirs(os.path.join(args.save_dir, 'save_results'))

    args_file = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')

    _logger = Logger(log_file_name=os.path.join(args.save_dir, args.log_file_name), 
        logger_name=args.logger_name).get_log()

    return _logger


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False

def PSNR(H_fuse, H_ref):
    #Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.sum((H_ref_reshaped-H_fuse_reshaped)**2, dim=1)/H_fuse_reshaped.shape[1])

    # Calculating max of H_ref for each band
    max_H_ref, _ = torch.max(H_ref_reshaped, dim=1)

    # Calculating PSNR
    PSNR = torch.nansum(10*torch.log10(torch.div(max_H_ref, rmse)**2))/N_spectral

    return PSNR

def calc_psnr(img1, img2):
    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
    diff = (img1 - img2) / 255.0
    diff[:,:,0] = diff[:,:,0] * 65.738 / 256.0
    diff[:,:,1] = diff[:,:,1] * 129.057 / 256.0
    diff[:,:,2] = diff[:,:,2] * 25.064 / 256.0

    diff = np.sum(diff, axis=2)
    mse = np.mean(np.power(diff, 2))
    return -10 * math.log10(mse)
    
  
def calc_ssim(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
        # the same outputs as MATLAB's
    border = 0
    img1_y = np.dot(img1, [65.738,129.057,25.064])/256.0+16.0
    img2_y = np.dot(img2, [65.738,129.057,25.064])/256.0+16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h-border, border:w-border]
    img2_y = img2_y[border:h-border, border:w-border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calc_psnr_and_ssim(sr, hr):
    ### args:
        # sr: pytorch tensor, range [-1, 1]
        # hr: pytorch tensor, range [-1, 1]

    ### prepare data
    sr = (sr+1.) * 127.5
    hr = (hr+1.) * 127.5
    if (sr.size() != hr.size()):
        h_min = min(sr.size(2), hr.size(2))
        w_min = min(sr.size(3), hr.size(3))
        sr = sr[:, :, :h_min, :w_min]
        hr = hr[:, :, :h_min, :w_min]

    img1 = np.transpose(sr.squeeze().round().cpu().numpy(), (1,2,0))
    img2 = np.transpose(hr.squeeze().round().cpu().numpy(), (1,2,0))

    psnr = calc_psnr(img1, img2)
    ssim = calc_ssim(img1, img2)

    return psnr, ssim

# def cal_ssim(sr,hr):
#     ### args:
#         # sr: pytorch tensor, range [-1, 1]
#         # hr: pytorch tensor, range [-1, 1]
#     hr = hr.cpu().numpy()
#     sr = sr.cpu().numpy()
#     # print(hr.shape)
#     # print(sr.shape)
#     Metric = iv.spectra_metric(hr,sr)
#     # if math.isinf(Metric.PSNR()):
#     #     print(sr)
#     # print(Metric.PSNR(), Metric.SSIM())
#     return Metric.SSIM()

#Cross-correlation matrix
def cross_correlation(H_fuse, H_ref):
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating mean value
    mean_fuse = torch.mean(H_fuse_reshaped, 1).unsqueeze(1)
    mean_ref = torch.mean(H_ref_reshaped, 1).unsqueeze(1)

    CC = torch.sum((H_fuse_reshaped- mean_fuse)*(H_ref_reshaped-mean_ref), 1)/torch.sqrt(torch.sum((H_fuse_reshaped- mean_fuse)**2, 1)*torch.sum((H_ref_reshaped-mean_ref)**2, 1))

    CC = torch.mean(CC)
    return CC


# Spectral-Angle-Mapper (SAM)
def SAM(H_fuse, H_ref):
    # Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)
    N_pixels = H_fuse_reshaped.shape[1]

    # Calculating inner product
    inner_prod = torch.nansum(H_fuse_reshaped * H_ref_reshaped, 0)
    fuse_norm = torch.nansum(H_fuse_reshaped ** 2, dim=0).sqrt()
    ref_norm = torch.nansum(H_ref_reshaped ** 2, dim=0).sqrt()

    # Calculating SAM
    SAM = torch.rad2deg(torch.nansum(torch.acos(inner_prod / (fuse_norm * ref_norm))) / N_pixels)
    return SAM

# Root-Mean-Squared Error (RMSE)
def RMSE(H_fuse, H_ref):
    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(-1)
    H_ref_reshaped = H_ref.view(-1)

    # Calculating RMSE
    RMSE = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped)**2)/H_fuse_reshaped.shape[0])
    return RMSE


# Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
def ERGAS(H_fuse, H_ref, beta):
    # Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)
    N_pixels = H_fuse_reshaped.shape[1]

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=1) / N_pixels)
    mu_ref = torch.mean(H_ref_reshaped, dim=1)

    # Calculating Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
    ERGAS = 100 * (1 / beta) * torch.sqrt(torch.nansum(torch.div(rmse, mu_ref) ** 2) / N_spectral)
    return ERGAS

def cal_ssim(im1,im2):
    assert im1.shape==im2.shape
    im1 = im1.cpu().numpy().astype(np.float64)
    im2 = im2.cpu().numpy().astype(np.float64)
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 1
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim