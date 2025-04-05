#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Before run this file, please ensure running <python -m visdom.server> in current environment.
Then, please go to http:localhost://#display_port# to see the visulizations.
"""
import torch
import time
import hues
import os
from data import get_dataloader
from model import create_model
from options.train_options import TrainOptions
from utils.visualizer import Visualizer,compute_psnr,compute_sam,compute_ergas,compute_rmse
import scipy.io as sio
import numpy as np
import random

def traintypechoose( traintype):
    if "spin" in traintype:
        angle = traintype.split('spin')[1]
        return "spin", angle
    elif "move" in traintype:
        pixel = traintype.split('move')[1]
        return "move", pixel
    elif "non" in traintype:
        pixel = traintype.split('non')[1]
        return "non", pixel
    elif "real" in traintype:
        return "real", None
    elif "well-regis" in traintype:
        return "well-regis", None
    else:
        raise Exception("please input the right traintype")
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1)

if __name__ == "__main__":

    train_opt = TrainOptions().parse()
    train_opt.name = 'CAVE_food' # 'CAVE_toy'
    train_opt.num_theta = 32
    train_opt.niter = 100
    train_opt.niter_decay = 4000
    # train_opt.lr = 4e-3
    train_opt.lr = 1e-3
    train_opt.lr_decay_iters = 200
    train_opt.display_port = 8097
#    train_opt.data_name    = 'CAVE'
#     train_opt.data_name    = 'CAVE'
#     train_opt.data_name = 'CAVE/wadc_normal_nonrigid2568_0_center'
    train_opt.data_name ='CAVE'
    #     train_opt.display_freq = 'CAVE/real_normal_nonrigid_0_center'
    train_opt.train_type = 'well-regis'
    typeoftrain, x = traintypechoose(train_opt.train_type)
    if typeoftrain in ["well-regis", "move", "spin"]:
        train_opt.srf_name     = 'paviaU_srf' # 'Landsat8_BGR'
        train_opt.mat_name     = 'pavia4CucaNet' # 'chart_and_stuffed_toy_ms'w
        # train_opt.mat_name = 'data'  # 'chart_and_stuffed_toy_ms'
    elif typeoftrain == "real":
        train_opt.srf_name     = 'Real_SRF'  # 'Landsat8_BGR'
        train_opt.mat_name     = 'data'  # 'chart_and_stuffed_toy_ms'
    elif typeoftrain == "non":
        train_opt.mat_name     = 'paviau_normal_nonrigid_'+str(1)+'_center/data' #非线性变换
        train_opt.srf_name = 'paviaU_srf'
    train_opt.scale_factor = 8
    train_opt.print_freq   = 10
    train_opt.save_freq    = 50
    train_opt.batchsize    = 1
    train_opt.which_epoch  = train_opt.niter + train_opt.niter_decay
    train_opt.useSoftmax   = 'No'
    train_opt.isCalSP = 'Yes'
    train_opt.display_port = 8097

    # trade-off parameters: could be better tuned
    # for auto-reconstruction
    train_opt.lambda_A = 10#L_res
    # train_opt.lambda_A = 10
    train_opt.lambda_B = 10 #B=G L_reg
    # gamma_1
    train_opt.lambda_C = train_opt.lambda_A
    # gamma_2
    # train_opt.lambda_G = 10
    train_opt.lambda_G = train_opt.lambda_B
    # alpha
    # train_opt.lambda_D = 0.1 #L_cons
    train_opt.lambda_D =0.1
    # beta
    train_opt.lambda_E =0.1 #L_spa0.01

    
    train_dataloader = get_dataloader(train_opt, isTrain=True)
    dataset_size = len(train_dataloader)
    train_model = create_model(train_opt, train_dataloader.hsi_channels,
                               # train_dataloader.msi_channels,
                               4,
                               train_dataloader.lrhsi_height,
                               train_dataloader.lrhsi_width,
                               train_dataloader.sp_matrix,
                               train_dataloader.sp_range)
    
    train_model.setup(train_opt)
    # visualizer = Visualizer(train_opt, train_dataloader.sp_matrix)
        
    total_steps = 0
    best_psnr = 0
    best_epoch = 0
    for epoch in range(train_opt.epoch_count, train_opt.niter + train_opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        train_psnr_list = []
        file = open("trainlog/trainlog_ss.txt", "a")
        for i, data in enumerate(train_dataloader):
            iter_start_time = time.time()
            total_steps += train_opt.batchsize
            epoch_iter += train_opt.batchsize
            # visualizer.reset()
            train_model.set_input(data, True)
            train_model.optimize_joint_parameters(epoch)

            hues.info("[{}/{} in {}/{}]".format(i,dataset_size//train_opt.batchsize,
                                                epoch,train_opt.niter + train_opt.niter_decay))
            train_psnr = train_model.cal_psnr()
            train_psnr_list.append(train_psnr)

            if epoch % train_opt.print_freq == 0:
                losses = train_model.get_current_losses()
                t = (time.time() - iter_start_time) / train_opt.batchsize
                # visualizer.print_current_losses(epoch, epoch_iter, losses, t)
                print("epoch:"+str(epoch)+" epoch_iter:"+str(epoch_iter)+"losses:"+str(losses)+" t:"+str(t))
                if train_opt.display_id > 0:
      # visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, train_opt, losses)
                    # visualizer.display_current_results(train_model.get_current_visuals(),
                    #                                    train_model.get_image_name(), epoch, True,
                    #                                    win_id=[1])
                    #
                    # visualizer.plot_spectral_lines(train_model.get_current_visuals(), train_model.get_image_name(),
                    #                                visual_corresponding_name=train_model.get_visual_corresponding_name(),
                    #                                win_id=[2,3])
                    # visualizer.plot_psnr_sam(train_model.get_current_visuals(), train_model.get_image_name(),
                    #                          epoch, float(epoch_iter) / dataset_size,
                    #                          train_model.get_visual_corresponding_name())
                    #
                    # visualizer.plot_lr(train_model.get_LR(), epoch)
                    visuals = train_model.get_current_visuals()
                    """psnr and sam updating with epoch"""
                    real_hsi = visuals["real_hhsi"].data.cpu().float().numpy()[0]
                    rec_hsi = (
                        visuals[train_model.get_visual_corresponding_name()["real_hhsi"]]
                        .data.cpu()
                        .float()
                        .numpy()[0]
                    )
                    # H = visuals["R"].data.cpu().numpy()
                    # stn = visuals["rec_RecoveredAbundance"].data.cpu().numpy()
                    lr_a = visuals["lr_abundance"].data.cpu().numpy()
                    result_erags = compute_ergas(real_hsi, rec_hsi,8)
                    result_rmse = compute_rmse(rec_hsi, real_hsi)
                    result_sam, _ = compute_sam(real_hsi, rec_hsi)
                    result_psnr,_ = compute_psnr(real_hsi, rec_hsi)
                    if result_psnr > best_psnr:
                        best_psnr = result_psnr
                        best_epoch = epoch
                    sio.savemat(os.path.join("./Results/", ''.join(data['name']) + '_well3.mat'),
                        {'recHSI': rec_hsi.transpose(1, 2, 0)})
                    # sio.savemat("generate_real_ttsr.mat",{'reconstructionLHSI':rec_hsi})
                    #     sio.savemat("TTSR_T_real.mat",{"T":H})
                    #     sio.savemat("STN.mat", {"STN": stn})
                    #     sio.savemat("lr_abundance.mat",{"lr_a":lr_a})
                    print("Best Epoch: "+str(best_epoch)+" "+str(best_psnr))
                    print ("Epoch:"+str(epoch)+" PSNR: "+str(result_psnr)+" SAM: "+str(result_sam)+" RMSE: "+str(result_rmse)+" ERAGS:"+str(result_erags))
                    # with open("trainlog.txt","w") as file:
                    file.write("Epoch:"+str(epoch)+" PSNR: "+str(result_psnr)+" SAM: "+str(result_sam)+" RMSE: "+str(result_rmse)+" ERAGS:"+str(result_erags)+'\n')
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.niter + train_opt.niter_decay, time.time() - epoch_start_time))
        train_model.update_learning_rate()
    file.write("Best Epoch:"+str(best_epoch)+" PSNR: "+str(best_psnr)+'\n')
    print("Best Epoch:"+str(best_epoch)+" PSNR: "+str(best_psnr))
    rec_hhsi=train_model.get_current_visuals()[train_model.get_visual_corresponding_name()['real_hhsi']].data.cpu().float().numpy()[0]

    
