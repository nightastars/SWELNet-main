import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from loader import *
from model.generator import Generator
from measure import compute_measure
import time
from losses import MixLoss, TVLossL1, L1_Charbonnier_loss, VGG_FeatureExtractor, VGGLoss
from torch.optim import lr_scheduler
import lpips
from sobel import functional_conv2d
import pyiqa
from utils import init_net
from torchmetrics.functional import structural_similarity_index_measure
from ema import update_ema_variables


class Solver(object):
    def __init__(self, args, data_loader, val_data_loader):
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max
        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu
        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.ema_decay = args.ema_decay

        # 评价指标
        self.niqe_metric = pyiqa.create_metric('niqe').to(self.device)
        self.nrqm_metric = pyiqa.create_metric('nrqm').to(self.device)
        self.pi_metric = pyiqa.create_metric('pi').to(self.device)

        self.G = Generator()
        self.ema_model = Generator()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.G = nn.DataParallel(self.G)
            self.ema_model = nn.DataParallel(self.ema_model)
        self.G.to(self.device)
        self.ema_model.to(self.device)

        self.update_ema = update_ema_variables

        self.mixloss = MixLoss().to(self.device)
        self.cbloss = L1_Charbonnier_loss().to(self.device)
        self.criterion = nn.L1Loss()
        self.tvlossL1 = TVLossL1().to(self.device)
        self.lpips_loss = lpips.LPIPS(net='vgg', verbose=False).to(self.device)  # squeeze
        self.vgg_feature = VGG_FeatureExtractor().to(self.device)
        self.vgg_loss = VGGLoss().to(self.device)

        self.lr_G = args.lr_G
        self.optimizer_G = optim.AdamW(self.G.parameters(), self.lr_G)
        self.schedulerG = lr_scheduler.CosineAnnealingLR(self.optimizer_G, self.num_epochs*len(self.data_loader), eta_min=1e-5)
        # self.schedulerG = lr_scheduler.StepLR(self.optimizer_G, step_size=self.decay_iters, gamma=0.5)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'generator_{}iter.ckpt'.format(iter_))
        torch.save(self.G.state_dict(), f)

    def load_model(self, net, iter_):
        f = os.path.join(self.save_path, 'generator_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            net.load_state_dict(state_d)
        else:
            net.load_state_dict(torch.load(f))

    def normalize_(self, image):
        image = (image - self.trunc_min) / (self.trunc_max - self.trunc_min)
        return image

    def denormalize1_(self, image):
        image = image * (self.trunc_max - self.trunc_min) + self.trunc_min
        return image

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))


    # train
    def train(self):
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        # 计算参数量
        G_param = self.get_parameter_number(self.G)
        print('G_param =', G_param)
        # self.load_model(self.G, 30 * 825)
        # self.load_model(self.ema_model, 30 * 825)
        # # init net
        # init_net(self.G)
        # 计时
        start_time = time.time()
        # 总的迭代次数
        total_iters = 0
        for epoch in range(1, self.num_epochs + 1):
            self.G.train(True)
            for iter_, (source, label, target) in enumerate(self.data_loader):  # target
                total_iters += 1

                source = self.normalize_(self.trunc(self.denormalize_(source)))
                label = self.normalize_(self.trunc(self.denormalize_(label)))
                target = self.normalize_(self.trunc(self.denormalize_(target)))

                # add 1 channel
                source = source.unsqueeze(0).float().to(self.device)
                label = label.unsqueeze(0).float().to(self.device)
                target = target.unsqueeze(0).float().to(self.device)

                # patch training
                if self.patch_size:
                    source = source.view(-1, 1, self.patch_size, self.patch_size)
                    label = label.view(-1, 1, self.patch_size, self.patch_size)
                    target = target.view(-1, 1, self.patch_size, self.patch_size)

                # Train the generator
                self.G.zero_grad()
                self.optimizer_G.zero_grad()

                # Sample images from generator   STUDENT model
                source_pred, target_Pseudo = self.G(source, target)

                # TEACHER model
                with torch.no_grad():
                    _, target_Pseudo_ema = self.ema_model(source, target)
                self.mylambda = self.sigmoid_rampup(total_iters, 0.3 * len(self.data_loader) * self.num_epochs / self.batch_size)
                T_loss = 1e-4 * self.mylambda * self.cbloss(target_Pseudo, target_Pseudo_ema.detach())  # 1e-4

                # Get generator loss
                source_loss = 1 * self.cbloss(source_pred, label) + 1e-5 * self.cbloss(functional_conv2d(source_pred), functional_conv2d(source))
                + 1e-3 * (1 - structural_similarity_index_measure(source_pred, label)) + 1e-5 * self.lpips_loss(source_pred, label).mean()
                target_loss = 1e-6 * self.tvlossL1(target_Pseudo) + 1e-5 * self.cbloss(functional_conv2d(target), functional_conv2d(target_Pseudo))
                # + 1.0 * self.lpips_loss(target, target_Pseudo).mean()  # 1e-2 * self.vgg_loss(target_Pseudo, target)  # 1e-2 * self.cbloss(self.vgg_feature(target_Pseudo), self.vgg_feature(target))   # 1e-6 1e-6

                totalloss_G = source_loss + target_loss + T_loss
                # Calculate gradients
                totalloss_G.backward()

                # optimize
                self.optimizer_G.step()
                self.update_ema(self.G, self.ema_model, self.ema_decay, total_iters)

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nlr: {:.10f} \t totalloss_G: {:.8f} \t source_loss: {:.8f} "
                          "\t target_loss: {:.12f} \t T_loss: {:.12f} \t TIME: {:.2f}s".format(total_iters, epoch,
                                                                      self.num_epochs, iter_ + 1,
                                                                      len(self.data_loader), self.optimizer_G.state_dict()['param_groups'][0]['lr'],
                                                                      totalloss_G.item(), source_loss.item(), target_loss.item(), target_loss.item(),
                                                                      time.time() - start_time))

                # learning rate decay
                self.schedulerG.step()

                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)

            # validate
            with torch.no_grad():
                pred_niqe_avg = 0
                self.G.eval()
                for i, (source, label, target) in enumerate(self.val_data_loader):
                    source = self.normalize_(self.trunc(self.denormalize_(source)))
                    label = self.normalize_(self.trunc(self.denormalize_(label)))
                    target = self.normalize_(self.trunc(self.denormalize_(target)))

                    source = source.unsqueeze(0).float().to(self.device)
                    label = label.unsqueeze(0).float().to(self.device)
                    target = target.unsqueeze(0).float().to(self.device)

                    source_pred, _ = self.G(source, target)

                    source = self.trunc(self.denormalize1_(source))
                    label = self.trunc(self.denormalize1_(label))
                    source_pred = self.trunc(self.denormalize1_(source_pred))
                    data_range = self.trunc_max - self.trunc_min
                    original_result, pred_result = compute_measure(source, label, source_pred, data_range)

            #         pred_niqe_avg += self.niqe_metric(target_pred)
            #
            # with open(self.save_path + '/pred_niqe_avg.txt', 'a') as f:
            #     f.write('EPOCH:%d loss:%.20f' % (epoch, pred_niqe_avg / len(self.val_data_loader)) + '\n')
            #     f.close()

                    pred_psnr_avg += pred_result[0]
                    pred_ssim_avg += pred_result[1]
                    pred_rmse_avg += pred_result[2]

            # 日志文件
            # with open(self.save_path+'/disc_loss.txt', 'a') as f:
            #     f.write('EPOCH:%d loss:%.20f' % (disc_loss) + '\n')
            #     f.close()

            with open(self.save_path+'/pred_psnr_avg.txt', 'a') as f:
                f.write('EPOCH:%d loss:%.20f' % (epoch, pred_psnr_avg / len(self.val_data_loader)) + '\n')
                f.close()

            with open(self.save_path+'/pred_ssim_avg.txt', 'a') as f:
                f.write('EPOCH:%d loss:%.20f' % (epoch, pred_ssim_avg / len(self.val_data_loader)) + '\n')
                f.close()

            with open(self.save_path+'/pred_rmse_avg.txt', 'a') as f:
                f.write('EPOCH:%d loss:%.20f' % (epoch, pred_rmse_avg / len(self.val_data_loader)) + '\n')
                f.close()
            pred_psnr_avg = 0
            pred_ssim_avg = 0
            pred_rmse_avg = 0

    # test
    def test(self):
        del self.G

        # load
        self.generator = Generator().to(self.device)
        self.load_model(self.generator, self.test_iters)

        # compute PSNR, SSIM, RMSE, std
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        ori_psnr_std, ori_ssim_std, ori_rmse_std = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        pred_psnr_std, pred_ssim_std, pred_rmse_std = 0, 0, 0
        ori_psnr_avg1, ori_ssim_avg1, ori_rmse_avg1 = [], [], []
        pred_psnr_avg1, pred_ssim_avg1, pred_rmse_avg1 = [], [], []

        # noisy_niqe_avg, clean_niqe_avg, pred_niqe_avg = 0, 0, 0
        # noisy_nrqm_avg, clean_nrqm_avg, pred_nrqm_avg = 0, 0, 0
        # noisy_pi_avg, clean_pi_avg, pred_pi_avg = 0, 0, 0
        #
        # noisy_niqe_std, clean_niqe_std, pred_niqe_std = 0, 0, 0
        # noisy_nrqm_std, clean_nrqm_std, pred_nrqm_std = 0, 0, 0
        # noisy_pi_std, clean_pi_std, pred_pi_std = 0, 0, 0
        #
        # noisy_niqe_avg_l, clean_niqe_avg_l, pred_niqe_avg_l = [], [], []
        # noisy_nrqm_avg_l, clean_nrqm_avg_l, pred_nrqm_avg_l = [], [], []
        # noisy_pi_avg_l, clean_pi_avg_l, pred_pi_avg_l = [], [], []

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            for i, (source, label, target) in enumerate(self.data_loader):

                shape_ = label.shape[-1]

                source = self.normalize_(self.trunc(self.denormalize_(source)))
                label = self.normalize_(self.trunc(self.denormalize_(label)))
                target = self.normalize_(self.trunc(self.denormalize_(target)))

                source = source.unsqueeze(0).float().to(self.device)
                label = label.unsqueeze(0).float().to(self.device)
                target = target.unsqueeze(0).float().to(self.device)

                source_pred, target_Pseudo = self.generator(source, target)
                # target_Pseudo = self.generator(target)

                # testtime
            torch.cuda.synchronize()
            end = time.time()

            #     x = self.trunc(self.denormalize1_(target.view(shape_, shape_).cpu().data.clamp(0, 1).detach()))
            #     y = self.trunc(self.denormalize1_(label.view(shape_, shape_).cpu().data.clamp(0, 1).detach()))
            #     pred = self.trunc(self.denormalize1_(target_Pseudo.view(shape_, shape_).cpu().data.clamp(0, 1).detach()))
            #
            #     np.save(os.path.join(self.save_path, 'x', '{}_result'.format(i)), x)
            #     np.save(os.path.join(self.save_path, 'y', '{}_result'.format(i)), y)
            #     np.save(os.path.join(self.save_path, 'pred', '{}_result'.format(i)), pred)
            #
            #     data_range = self.trunc_max - self.trunc_min
            #
            #     original_result, pred_result = compute_measure(x, y, pred, data_range)
            #     ori_psnr_avg += original_result[0]
            #     ori_psnr_avg1.append(original_result[0])
            #     ori_ssim_avg += original_result[1]
            #     ori_ssim_avg1.append(original_result[1])
            #     ori_rmse_avg += original_result[2]
            #     ori_rmse_avg1.append(original_result[2])
            #     pred_psnr_avg += pred_result[0]
            #     pred_psnr_avg1.append(pred_result[0])
            #     pred_ssim_avg += pred_result[1]
            #     pred_ssim_avg1.append(pred_result[1])
            #     pred_rmse_avg += pred_result[2]
            #     pred_rmse_avg1.append(pred_result[2])
            #
            #     # save result figure
            #     if self.result_fig:
            #         self.save_fig(x, y, pred, i, original_result, pred_result)
            #
            # # # testtime
            # # torch.cuda.synchronize()
            # # end = time.time()
            #
            # # calculate STD
            # for i in range(len(self.data_loader)):
            #     ori_psnr_std += (ori_psnr_avg1[i] - ori_psnr_avg / len(self.data_loader))**2
            #     ori_ssim_std += (ori_ssim_avg1[i] - ori_ssim_avg / len(self.data_loader))**2
            #     ori_rmse_std += (ori_rmse_avg1[i] - ori_rmse_avg / len(self.data_loader))**2
            #
            #     pred_psnr_std += (pred_psnr_avg1[i] - pred_psnr_avg / len(self.data_loader))**2
            #     pred_ssim_std += (pred_ssim_avg1[i] - pred_ssim_avg / len(self.data_loader))**2
            #     pred_rmse_std += (pred_rmse_avg1[i] - pred_rmse_avg / len(self.data_loader))**2
            #
            # # testtime
            # torch.cuda.synchronize()
            # end = time.time()
            #
            # print('\n')
            # print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} \nPSNR std: {:.4f} \nSSIM std: {:.4f} \nRMSE std: {:.4f}'.format(
            #     ori_psnr_avg / len(self.data_loader), ori_ssim_avg / len(self.data_loader), ori_rmse_avg / len(self.data_loader),
            #     pow(ori_psnr_std / len(self.data_loader), 0.5), pow(ori_ssim_std / len(self.data_loader), 0.5), pow(ori_rmse_std / len(self.data_loader), 0.5)))
            # print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} \nPSNR std: {:.4f} \nSSIM std: {:.4f} \nRMSE std: {:.4f}'.format(
            #     pred_psnr_avg / len(self.data_loader), pred_ssim_avg / len(self.data_loader), pred_rmse_avg / len(self.data_loader),
            #     pow(pred_psnr_std / len(self.data_loader), 0.5), pow(pred_ssim_std / len(self.data_loader), 0.5), pow(pred_rmse_std / len(self.data_loader), 0.5)))
            # print('\n')
            print('Test time: {:.4f} s'.format(end-start))

            # # calculate STD
            # for i in range(len(self.data_loader)):
            #     noisy_niqe_std += (noisy_niqe_avg_l[i] - noisy_niqe_avg / len(self.data_loader)) ** 2
            #     pred_niqe_std += (pred_niqe_avg_l[i] - pred_niqe_avg / len(self.data_loader)) ** 2
            #     clean_niqe_std += (clean_niqe_avg_l[i] - clean_niqe_avg / len(self.data_loader)) ** 2
            #
            #     noisy_nrqm_std += (noisy_nrqm_avg_l[i] - noisy_nrqm_avg / len(self.data_loader)) ** 2
            #     pred_nrqm_std += (pred_nrqm_avg_l[i] - pred_nrqm_avg / len(self.data_loader)) ** 2
            #     clean_nrqm_std += (clean_nrqm_avg_l[i] - clean_nrqm_avg / len(self.data_loader)) ** 2
            #
            #     noisy_pi_std += (noisy_pi_avg_l[i] - noisy_pi_avg / len(self.data_loader)) ** 2
            #     pred_pi_std += (pred_pi_avg_l[i] - pred_pi_avg / len(self.data_loader)) ** 2
            #     clean_pi_std += (clean_pi_avg_l[i] - clean_pi_avg / len(self.data_loader)) ** 2

            # # testtime
            # torch.cuda.synchronize()
            # # end = time.time()

            # print('\n')
            # print(
            #     'Low quality\nNIQE avg: {:.4f} \nNRQM avg: {:.4f} \nPI avg: {:.4f} \nNIQE std: {:.4f} \nSSIM std: {:.4f} \nPI std: {:.4f}'.format(
            #         noisy_niqe_avg / len(self.data_loader), noisy_nrqm_avg / len(self.data_loader),
            #         noisy_pi_avg / len(self.data_loader),
            #         pow(noisy_niqe_std / len(self.data_loader), 0.5), pow(noisy_nrqm_std / len(self.data_loader), 0.5),
            #         pow(noisy_pi_std / len(self.data_loader), 0.5)))
            # print(
            #     'pred\nNIQE avg: {:.4f} \nNRQM avg: {:.4f} \nPI avg: {:.4f} \nNIQE std: {:.4f} \nSSIM std: {:.4f} \nPI std: {:.4f}'.format(
            #         pred_niqe_avg / len(self.data_loader), pred_nrqm_avg / len(self.data_loader),
            #         pred_pi_avg / len(self.data_loader),
            #         pow(pred_niqe_std / len(self.data_loader), 0.5), pow(pred_nrqm_std / len(self.data_loader), 0.5),
            #         pow(pred_pi_std / len(self.data_loader), 0.5)))
            # print(
            #     'High quality\nNIQE avg: {:.4f} \nNRQM avg: {:.4f} \nPI avg: {:.4f} \nNIQE std: {:.4f} \nSSIM std: {:.4f} \nPI std: {:.4f}'.format(
            #         clean_niqe_avg / len(self.data_loader), clean_nrqm_avg / len(self.data_loader),
            #         clean_pi_avg / len(self.data_loader),
            #         pow(clean_niqe_std / len(self.data_loader), 0.5), pow(clean_nrqm_std / len(self.data_loader), 0.5),
            #         pow(clean_pi_std / len(self.data_loader), 0.5)))
            # print('\n')
            # print('Test time: {:.4f} s'.format(end - start))

        # '''PSNR,SSIM,RMSE IMAGES'''
        # fig = plt.figure()
        # plt.title('PSNR')
        # ax1 = fig.add_subplot(1, 1, 1)
        # ax1.plot(list(range(len(self.data_loader))), ori_psnr_avg1, label='ori_psnr')
        # ax1.plot(list(range(len(self.data_loader))), pred_psnr_avg1, label='pred_psnr')
        # plt.legend()  # 显示图例
        # plt.xlabel('image_num')
        # plt.ylabel('psnr')
        # plt.savefig('psnr.png')
        #
        # fig = plt.figure()
        # plt.title('SSIM')
        # ax2 = fig.add_subplot(1, 1, 1)
        # ax2.plot(list(range(len(self.data_loader))), ori_ssim_avg1, label='ori_ssim')
        # ax2.plot(list(range(len(self.data_loader))), pred_ssim_avg1, label='pred_ssim')
        # plt.legend()  # 显示图例
        # plt.xlabel('image_num')
        # plt.ylabel('ssim')
        # plt.savefig('ssim.png')
        #
        # fig = plt.figure()
        # plt.title('RMSE')
        # ax3 = fig.add_subplot(1, 1, 1)
        # ax3.plot(list(range(len(self.data_loader))), ori_rmse_avg1, label='ori_rmse')
        # ax3.plot(list(range(len(self.data_loader))), pred_rmse_avg1, label='pred_rmse')
        # plt.legend()  # 显示图例
        # plt.xlabel('image_num')
        # plt.ylabel('rmse')
        # plt.savefig('rmse.png')
        # plt.show()
