import argparse
from Train import Train, Init, generate
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--c_nums", type=int, default=10)
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--img_w", type=int, default=64)
    parser.add_argument("--img_c", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--train_itr", type=int, default=100000)
    parser.add_argument("--path_dataset", type=str, default="./dataset//")
    parser.add_argument("--path_save_img", type=str, default="./save_img//")
    parser.add_argument("--path_save_para", type=str, default="./save_para//")
    parser.add_argument("--path_results", type=str, default="./results//")
    parser.add_argument("--is_trained", type=bool, default=False)

    args = parser.parse_args()

    if not args.is_trained:
        Train(args.batch_size, args.z_dim, args.c_nums, args.img_h, args.img_w, args.img_c, args.lr, args.beta1, args.beta2, args.train_itr, args.path_dataset, args.path_save_img, args.path_save_para)
    else:
        parser.add_argument("--label1", type=int, default=0)
        parser.add_argument("--label2", type=int, default=1)
        parser.add_argument("--alpha", type=float, default=0.5)
        args = parser.parse_args()
        target, sess, z, train_phase, y1, y2, alpha = Init()
        z_np = np.random.normal(0, 1, [1, args.z_dim])
        for a in range(11):
            args.alpha = a / 10
            generate(z_np, args.path_results, args.label1, args.label2, args.alpha, target, sess, z, train_phase, y1, y2, alpha)