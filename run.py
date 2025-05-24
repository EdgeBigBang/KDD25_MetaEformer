import random
import numpy as np
import argparse
import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), ''))

from exp import train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data process
    parser.add_argument('--task_name', type=str, default='ECL_pic_hight', help='')
    parser.add_argument('--dataset', type=str, default='ECL', help='Data class for evaluation')
    parser.add_argument('--data', type=str, default='ECL_270.csv', help='Data class for evaluation')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument("--minmax_scaler", "-mm", action="store_true", default=True)

    # random seed
    parser.add_argument('--random_seed', type=int, default=2, help='random seed')

    # train setting
    parser.add_argument("--num_epoches", "-e", type=int, default=150)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=256)
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task following by Informer')


    # model setting
    parser.add_argument("--e_layers", "-nel", type=int, default=1)
    parser.add_argument("--d_layers", "-ndl", type=int, default=1)
    parser.add_argument("--d_model", "-dm", type=int, default=256)
    parser.add_argument("--d_low", "-dlow", type=int, default=10)
    parser.add_argument("--n_heads", "-nh", type=int, default=8)
    parser.add_argument("--d_ff", "-hs", type=int, default=256)
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument("--label_len", "-dl", type=int, default=12)
    parser.add_argument("--pred_len", "-ol", type=int, default=24)
    parser.add_argument("--enc_len", "-not", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default='gelu')

    parser.add_argument("--dim_static", type=int, default=12)

    parser.add_argument("--wave_class", type=int, default=350, help='The size of meta-pattern pool')
    parser.add_argument("--wave_len", type=int, default=16, help='The length of waves')
    parser.add_argument("--low_dim", type=int, default=10)

    parser.add_argument("--if_padding", type=bool, default=True)
    parser.add_argument("--mpp_update", type=int, default=50)
    parser.add_argument("--kernel_size", type=int, default=24)
    parser.add_argument("--sim_num", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1)

    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument("-output_attention", type=bool, default=False)
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')

    # other settings
    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--save_model", "-sm", type=bool, default=True)
    parser.add_argument("--load_model", "-lm", type=bool, default=False)
    parser.add_argument("--show_plot", "-sp", type=bool, default=False)

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')

    args = parser.parse_args()

    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    if args.run_test:
        torch.autograd.set_detect_anomaly(True)

        losses, test_losses, mse_l, mae_l = train(args)
        torch.cuda.empty_cache()
        input("over please press Enter")

    else:
        test(args)
        torch.cuda.empty_cache()
