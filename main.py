import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
import torch


def main(args):
    cudnn.benchmark = True
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        x_path = os.path.join(args.save_path, 'x')
        y_path = os.path.join(args.save_path, 'y')
        pred_path = os.path.join(args.save_path, 'pred')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
        if not os.path.exists(x_path):
            os.makedirs(x_path)
            print('Create path : {}'.format(x_path))
        if not os.path.exists(y_path):
            os.makedirs(y_path)
            print('Create path : {}'.format(y_path))
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
            print('Create path : {}'.format(pred_path))
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    val_data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             augment=False,
                             saved_path=args.validata_path,
                             test_patient=args.test_patient if args.mode == 'train' else args.val_patient,
                             patch_n=None,
                             patch_size=None,
                             transform=args.transform,
                             batch_size=1,
                             num_workers=args.num_workers)

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             augment=args.augment if args.mode == 'train' else False,
                             saved_path=args.train_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader, val_data_loader)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help="train | test")
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--augment', type=bool, default=False)
    parser.add_argument('--train_path', type=str, default=r'/data/wangjiping/semi-supervised-CT/semi-supervised-dataset1/test/')
    parser.add_argument('--validata_path', type=str, default=r'/data/wangjiping/semi-supervised-CT/semi-supervised-dataset1/val/')
    # parser.add_argument('--train_path', type=str, default=r'/data/wangjiping/semi-supervised-CT/semi-supervised-dataset-mingfeng/test/')
    # parser.add_argument('--validata_path', type=str, default=r'/data/wangjiping/semi-supervised-CT/semi-supervised-dataset-mingfeng/val/')
    parser.add_argument('--save_path', type=str, default='./save1-mayo-3gerecu/')  # 18
    parser.add_argument('--test_patient', type=str, default='TEST')
    parser.add_argument('--val_patient', type=str, default='VAL')
    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-300.0)
    parser.add_argument('--trunc_max', type=float, default=300.0)
    parser.add_argument('--transform', type=bool, default=False)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--print_iters', type=int, default=40)
    parser.add_argument('--decay_iters', type=int, default=853*20)
    parser.add_argument('--save_iters', type=int, default=853)
    parser.add_argument('--test_iters', type=int, default=853*200)  # TV-188
    parser.add_argument('--lr_G', type=float, default=2e-4)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=48)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    main(args)

