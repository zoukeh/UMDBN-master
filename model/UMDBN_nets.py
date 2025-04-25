#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 双线性模型
import torch
import torch.nn
from torch.autograd import Variable
import itertools
import model.network_mulitscale0 as network
from .base_model import BaseModel
import numpy as np
from .NonLinear_Transformer import Transformer


class UMDBNnets(BaseModel):
    def name(self):
        return 'UMDBNets'

    @staticmethod
    def modify_commandline_options(parser, isTrain=True):

        parser.set_defaults(no_dropout=True)
        if isTrain:
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for lr_lr')
            parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for msi_msi    beta')
            parser.add_argument('--lambda_C', type=float, default=1.0, help='weight for msi_s_lr   alpha')
            parser.add_argument('--lambda_D', type=float, default=1.0, help='weight for sum2one    mu')
            parser.add_argument('--lambda_E', type=float, default=1.0, help='weight for sparse     nu')
            parser.add_argument('--n_res', type=int, default=3)
            parser.add_argument('--avg_crite', type=str, default='No')
            parser.add_argument('--isCalSP', type=str, default='Yes')
            parser.add_argument("--useSoftmax", type=str, default='Yes')
        return parser

    def initialize(self, opt, hsi_channels, msi_channels, lrhsi_hei, lrhsi_wid, sp_matrix, sp_range):
        BaseModel.initialize(self, opt)
        self.opt = opt
        # hsi_channels = 102
        # msi_channels = 6

        # sp_range = np.array([[0, 107],[0, 107],[0, 107]])
        self.visual_names = ['real_lhsi', 'rec_lrhsi']
        num_s = self.opt.num_theta  # 端元数量
        ngf = 64
        lr_size = 32
        # net getnerator
        # hsi
        self.net_bl_lr = network.define_bm_net(temperature=16, gpu_ids= self.gpu_ids)  # 双线性 MTM
        self.net_reclr = network.define_s2img_2stream(input_c1 = num_s, input_tf = self.net_bl_lr, output_ch=hsi_channels,
                                                      gpu_ids= self.gpu_ids)
        self.net_MultiScale = network.define_multiscaleNet(hsi_channels=hsi_channels, msi_channels=msi_channels, n_feat=num_s ,gpu_ids=self.gpu_ids)
        self.net_PANet = network.define_PANet(in_channels=num_s, out_channels=num_s, gpu_ids= self.gpu_ids)
        # self.net_reconstruct_hrhsi = network.define_s2img(input_ch=num_s, output_ch=msi_channels, gpu_ids=self.gpu_ids)
        self.net_bl_msi = network.define_bm_net(temperature=8, gpu_ids=self.gpu_ids)
        self.net_recmsi = network.define_s2img_2stream_msi(input_c1= num_s, input_tf=self.net_bl_msi, output_ch=msi_channels,
                                                       gpu_ids=self.gpu_ids)

        self.net_PSF = network.define_psf(scale=opt.scale_factor, gpu_ids=self.gpu_ids)  # 生成的HRHSI下采样为Lrhsi
        self.net_PSF_2 = network.define_psf_2(scale=opt.scale_factor, gpu_ids=self.gpu_ids)  # 用于计算互相关信息时
        self.net_PSF_3 = network.define_psf_2(scale=opt.scale_factor, gpu_ids=self.gpu_ids)  # 用于Merge
        self.net_Merge = network.define_Merge(input_c=num_s,out_c=num_s,gpu_ids=self.gpu_ids)  # 用于计算变形场时
        # HRHSI to HRMSI
        self.net_HR2MSI = network.define_hr2msi(args=self.opt, hsi_channels=hsi_channels, msi_channels=msi_channels,
                                                sp_matrix=sp_matrix, sp_range=sp_range, gpu_ids=self.gpu_ids)
        # cross
        self.net_mut_spa = network.define_spatial_AM(input_ch=ngf, kernel_sz=3, gpu_ids=self.gpu_ids)  # 和net_LR_1输出通道一样
        self.net_mut_spe = network.define_spectral_AM(input_ch=ngf, input_hei=int(lrhsi_hei), input_wid=int(lrhsi_wid),
                                                      gpu_ids=self.gpu_ids)
        # LOSS
        if self.opt.avg_crite == "No":
            self.criterionL1Loss = torch.nn.L1Loss(size_average=False).to(self.device)
        else:
            self.criterionL1Loss = torch.nn.L1Loss(size_average=True).to(self.device)

        self.criterionPixelwise = self.criterionL1Loss
        self.criterionCrossImformation = network.LossPixlwise().to(self.device)
        self.criterionSumToOne = network.SumToOneLoss().to(self.device)
        self.criterionSparse = network.SparseKLloss().to(self.device)

        self.model_names = ['MultiScale','PANet',
                            'PSF', 'HR2MSI', 'PSF_2', 'PSF_3','mut_spa', 'mut_spe',
                            'recmsi', 'reclr',  'bl_lr', 'bl_msi']
        self.setup_optimizers()
        self.visual_corresponding_name = {}

    def setup_optimizers(self, lr=None):
        if lr == None:
            lr = self.opt.lr
        else:
            isinstance(lr, float)
            lr = lr
        self.optimizers = []
        # 0.5
        self.optimizer_multiscale = torch.optim.Adam(itertools.chain(self.net_MultiScale.parameters()), lr=lr * 0.5,
                                                betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_multiscale)
        self.optimizer_panet = torch.optim.Adam(itertools.chain(self.net_PANet.parameters()), lr=lr * 0.5,
                                                betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_panet)

        self.optimizers_G_reclr = torch.optim.Adam(itertools.chain(self.net_reclr.parameters()), lr=lr,
                                                   betas=(0.9, 0.999))
        self.optimizers.append(self.optimizers_G_reclr)
        self.optimizers_G_recmsi = torch.optim.Adam(itertools.chain(self.net_recmsi.parameters()), lr=lr,
                                                    betas=(0.9, 0.999))
        self.optimizers.append(self.optimizers_G_recmsi)
        # 0.2
        self.optimizer_PSF = torch.optim.Adam(itertools.chain(self.net_PSF.parameters()), lr=lr * 0.2,
                                              betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_PSF)
        self.optimizer_PSF_2 = torch.optim.Adam(itertools.chain(self.net_PSF_2.parameters()), lr=lr * 0.2,
                                                betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_PSF_2)
        self.optimizer_Merge = torch.optim.Adam(itertools.chain(self.net_Merge.parameters()), lr=lr * 0.2,
                                                betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_Merge)
        self.optimizer_PSF_3 = torch.optim.Adam(itertools.chain(self.net_PSF_3.parameters()), lr=lr * 0.2,
                                                betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_PSF_3)
        self.optimizer_mut_spa = torch.optim.Adam(itertools.chain(self.net_mut_spa.parameters()), lr=lr * 0.2,
                                                  betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_mut_spa)
        self.optimizer_mut_spe = torch.optim.Adam(itertools.chain(self.net_mut_spe.parameters()), lr=lr * 0.2,
                                                  betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_mut_spe)

        if self.opt.isCalSP == 'Yes':
            # 0.2
            self.optimizer_HR2MSI = torch.optim.Adam(itertools.chain(self.net_HR2MSI.parameters()), lr=lr * 0.2,
                                                     betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_HR2MSI)

    def set_input(self, input, isTrain=True):
        if isTrain:
            self.real_lhsi = Variable(input['lhsi'], requires_grad=True).to(self.device)
            self.real_hmsi = Variable(input['hmsi'], requires_grad=True).to(self.device)
            self.real_hhsi = Variable(input['hhsi'], requires_grad=True).to(self.device)

        else:
            with torch.no_grad():
                self.real_lhsi = Variable(input['lhsi'], requires_grad=False).to(self.device)
                self.real_hmsi = Variable(input['hmsi'], requires_grad=False).to(self.device)
                self.real_hhsi = Variable(input['hhsi'], requires_grad=False).to(self.device)

        self.image_name = input['name']
        self.real_input = input

    def my_forward(self):
        # LrHSI, HrMSI to themselves

        self.lr_abundance,self.msi_abundance,self.d1 ,self.d2, self.d3, self.d4 = self.net_MultiScale(self.real_lhsi,self.real_hmsi)
        # self.lr_abundance, self.msi_abundance, self.d1, self.d2, self.d3 = self.net_MultiScale(self.real_lhsi,self.real_hmsi)

        self.aff_lrhsi = self.net_PANet(self.lr_abundance, self.d1 , self.d2, self.d3,self.d4)

        self.rec_lrhsi,self.lr_abundance_bi = self.net_reclr(self.lr_abundance, None)  # 重构的lrhsi
        self.rec_aff_lrhsi,_= self.net_reclr(self.aff_lrhsi, None)

        self.rec_hrmsi, self.msi_abundance_bi = self.net_recmsi(self.msi_abundance, None)  # 重构的HrMSI

        # HrHSI to LrHSI, HrMSI
        self.rec_hrhsi,_ = self.net_reclr(self.msi_abundance,self.msi_abundance_bi)  # 生成的HRHSI
        self.rec_hrhsi2lrhsi = self.net_PSF(self.rec_hrhsi)  # 生成的HRHSI下采样为Lrhsi
        self.rec_hrhsi2hrmsi = self.net_HR2MSI(self.rec_hrhsi)  # 生成的HRHSI光谱下采样为HRMSI

        # LrHSI, HrMSI to LrMSI
        self.rec_lrhsi_lrmsi = self.net_HR2MSI(self.real_lhsi)  #
        self.rec_hrmsi_lrmsi = self.net_PSF(self.real_hmsi)  #

        self.visual_corresponding_name['real_lhsi'] = 'rec_lrhsi'
        self.visual_corresponding_name['real_hmsi'] = 'rec_hrmsi'
        self.visual_corresponding_name['real_hhsi'] = 'rec_hrhsi'

    def my_backward_g_joint(self, epoch):
        # lr-1
        self.loss_lr_pixelwise = self.criterionPixelwise(self.real_lhsi, self.rec_lrhsi) * self.opt.lambda_A
        self.loss_lr_s_sumtoone = self.criterionSumToOne([self.lr_abundance,self.lr_abundance_bi]) * self.opt.lambda_D
        #        self.loss_lr_s_sumtoone = self.criterionSumToOne(self.net_reclr) * self.opt.lambda_D
        self.loss_lr_sparse = self.criterionSparse(self.lr_abundance) * self.opt.lambda_E
        self.loss_lr = self.loss_lr_pixelwise + self.loss_lr_s_sumtoone + self.loss_lr_sparse
        # self.loss_lr = self.loss_lr_pixelwise + self.loss_lr_s_sumtoone
        # lr-2: PSF
        # self.loss_msi_ss_lr =  self.criterionPixelwise(self.real_lhsi, self.rec_hrhsi2lrhsi) * self.opt.lambda_G
        self.loss_msi_ss_lr = self.criterionPixelwise(self.rec_aff_lrhsi, self.rec_hrhsi2lrhsi) *  self.opt.lambda_B

        # msi-1
        self.loss_msi_pixelwise = self.criterionPixelwise(self.real_hmsi, self.rec_hrmsi) * self.opt.lambda_B
        self.loss_msi_s_sumtoone = self.criterionSumToOne([self.msi_abundance,self.msi_abundance_bi]) * self.opt.lambda_D
        # self.loss_abundance_asc =
        #        self.loss_msi_s_sumtoone = self.criterionSumToOne(self.net_recmsi) * self.opt.lambda_D
        self.loss_msi_sparse = self.criterionSparse(self.msi_abundance) * self.opt.lambda_E
        self.loss_msi = self.loss_msi_pixelwise + self.loss_msi_s_sumtoone + self.loss_msi_sparse
        # self.loss_msi = self.loss_msi_pixelwise + self.loss_msi_s_sumtoone
        # msi-2: SRF
        self.loss_msi_ss_msi = self.criterionPixelwise(self.real_hmsi, self.rec_hrhsi2hrmsi) * self.opt.lambda_A
        # lrmsi
        self.loss_lrmsi_pixelwise = self.criterionCrossImformation(self.rec_lrhsi_lrmsi,
                                                            self.rec_hrmsi_lrmsi) * self.opt.lambda_A

        # self.loss_joint = self.loss_lr + self.loss_msi + self.loss_msi_ss_lr + self.loss_msi_ss_msi + self.loss_lrmsi_pixelwise
        self.loss_joint = self.loss_lr + self.loss_msi + self.loss_msi_ss_lr + self.loss_msi_ss_msi + self.loss_lrmsi_pixelwise
        self.loss_joint.backward(retain_graph=False)

    def optimize_joint_parameters(self, epoch):
        # self.loss_names = ["lr_pixelwise", 'lr_s_sumtoone', 'lr_sparse', 'lr',
        #                    'msi_pixelwise', 'msi_s_sumtoone', 'msi_sparse', 'msi',
        #                    'msi_ss_lr', 'lrmsi_pixelwise']
        self.loss_names = ["lr_pixelwise", 'lr_s_sumtoone', 'lr',
                           'msi_pixelwise', 'msi_s_sumtoone',  'msi',
                           'msi_ss_lr', 'lrmsi_pixelwise']
        # self.visual_names = ['real_lhsi', 'rec_lrhsi', 'real_hmsi', 'rec_hrmsi', 'real_hhsi', 'rec_hrhsi','rec_RecoveredAbundance','lr_abundance']
        self.visual_names = ['real_lhsi', 'rec_lrhsi', 'real_hmsi', 'rec_hrmsi', 'real_hhsi', 'rec_hrhsi',
                              'lr_abundance']
        self.set_requires_grad([self.net_MultiScale,self.net_PANet, self.net_PSF, self.net_HR2MSI,
                                self.net_PSF_2, self.net_mut_spa, self.net_mut_spe, self.net_recmsi, self.net_reclr,
                                self.net_Merge], True)
        self.my_forward()

        self.optimizer_multiscale.zero_grad()
        self.optimizer_panet.zero_grad()

        self.optimizer_PSF.zero_grad()
        self.optimizer_PSF_2.zero_grad()
        self.optimizer_PSF_3.zero_grad()
        self.optimizer_Merge.zero_grad()
        self.optimizers_G_reclr.zero_grad()
        self.optimizers_G_recmsi.zero_grad()
        # self.optimizers_G_rechsi.zero_grad()
        self.optimizer_mut_spa.zero_grad()
        self.optimizer_mut_spe.zero_grad()
        # self.optimizer_ms2img
        if self.opt.isCalSP == 'Yes':
            self.optimizer_HR2MSI.zero_grad()

        self.my_backward_g_joint(epoch)
        self.optimizer_multiscale.step()
        self.optimizer_panet.step()
        self.optimizer_PSF.step()
        self.optimizer_PSF_2.step()
        self.optimizer_PSF_3.step()
        self.optimizer_Merge.step()
        self.optimizers_G_recmsi.step()
        self.optimizers_G_reclr.step()
        self.optimizer_mut_spa.step()
        self.optimizer_mut_spe.step()
        # self.optimizers_G_rechsi.step()
        if self.opt.isCalSP == 'Yes':
            self.optimizer_HR2MSI.step()

        cliper_zeroone = network.ZeroOneClipper()
        self.net_reclr.apply(cliper_zeroone)
        self.net_recmsi.apply(cliper_zeroone)
        # self.net_rechsi.apply(cliper_zeroone)
        if self.opt.isCalSP == 'Yes':
            cliper_sumtoone = network.SumToOneClipper()
            self.net_HR2MSI.apply(cliper_sumtoone)

    def get_visual_corresponding_name(self):
        return self.visual_corresponding_name

    def cal_psnr(self):
        real_hsi = self.real_hhsi.data.cpu().float().numpy()[0]
        rec_hsi = self.rec_hrhsi.data.cpu().float().numpy()[0]
        return self.compute_psnr(real_hsi, rec_hsi)

    def compute_psnr(self, img1, img2):
        assert img1.ndim == 3 and img2.ndim == 3

        img_c, img_w, img_h = img1.shape
        ref = img1.reshape(img_c, -1)
        tar = img2.reshape(img_c, -1)
        msr = np.mean((ref - tar) ** 2, 1)
        max2 = np.max(ref) ** 2
        psnrall = 10 * np.log10(max2 / msr)
        out_mean = np.mean(psnrall)
        return out_mean

    def get_LR(self):
        lr = self.optimizers[0].param_groups[0]['lr'] * 2 * 1000
        return lr



