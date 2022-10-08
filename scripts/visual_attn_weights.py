import pickle
import numpy as np
import torch
import argparse, logging
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
path = r"/home/think/IKD-mmt/"


def main(args):
    # save_dir = '/home/think/checkpoints/multi30k/en-de/inverseKD_res_m_l2_1enc'
    attn_f = open(args.save_dir + f'/AttnWeight.pickle', 'rb')

    # get the pratice line sort
    line_nums = np.zeros(shape =[1000], dtype=int)
    with open(args.save_dir + f'/line_num.txt', 'r') as line_num_f:
        lines = line_num_f.readlines()
        for id, l in enumerate(lines):
            num_i = int(l.strip())
            line_nums[id] = num_i
    line_nums = line_nums.tolist()

    # load numpy from .pkl file
    attn_out = []
    while True:
        try:
            attn_i = pickle.load(attn_f)
            attn_out.append(attn_i)
        except EOFError:
            break

    attn_out = [x for _,x in sorted(zip(line_nums, attn_out))]#attn_out[]
    # print()

    # two case visualization
    # case a : a ballerina in blue twir@@ ls . <eos>
    # 首先把textual部分删掉，最后<eos>删掉, 然后把twir@@ ls的数值加起来了
    attn_w_a = torch.from_numpy(attn_out[77])[:,:49, :-1]
    attn_w_a_last= attn_w_a[:,:49, -1] + attn_w_a[:,:49, -2]
    attn_w_a_prev = attn_w_a[:,:49, :-2]
    attn_w_a = torch.cat((attn_w_a_prev, attn_w_a_last.unsqueeze(-1)), dim=2)

    # Display matrix
    plt.imshow(attn_w_a[0], cmap='Blues', interpolation='nearest', origin='lower')
    plt.xticks((np.arange(0, 6, 1)))
    plt.yticks((np.arange(0, 50, 7)))
    plt.savefig(path + r"case_a_h1.png", bbox_inches="tight", pad_inches=0)
    plt.imshow(attn_w_a[1], cmap='Blues', interpolation='nearest', origin='lower')
    plt.xticks((np.arange(0, 6, 1)))
    plt.yticks((np.arange(0, 50, 7)))
    plt.savefig(path + r"case_a_h2.png", bbox_inches="tight", pad_inches=0)
    plt.imshow(attn_w_a[2], cmap='Blues', interpolation='nearest', origin='lower')
    plt.xticks((np.arange(0, 6, 1)))
    plt.yticks((np.arange(0, 50, 7)))
    plt.savefig(path + r"case_a_h3.png", bbox_inches="tight", pad_inches=0)
    plt.imshow(attn_w_a[3], cmap='Blues', interpolation='nearest', origin='lower')
    plt.xticks((np.arange(0, 6, 1)))
    plt.yticks((np.arange(0, 50, 7)))
    plt.savefig(path + r"case_a_h4.png", bbox_inches="tight", pad_inches=0)


    # case b: a little girl is walking barefoot on the sand .
    attn_w_b = torch.from_numpy(attn_out[652])[:,:49, :-1]

    # Display matrix
    plt.imshow(attn_w_b[0], cmap='Blues', interpolation='nearest', origin='lower')
    plt.xticks((np.arange(0, 10, 1)))
    plt.yticks((np.arange(0, 50, 7)))
    plt.savefig(path + r"case_b_h1.png", bbox_inches="tight", pad_inches=0)
    plt.imshow(attn_w_b[1], cmap='Blues', interpolation='nearest', origin='lower')
    plt.xticks((np.arange(0, 10, 1)))
    plt.yticks((np.arange(0, 50, 7)))
    plt.savefig(path + r"case_b_h2.png", bbox_inches="tight", pad_inches=0)
    plt.imshow(attn_w_b[2], cmap='Blues', interpolation='nearest', origin='lower')
    plt.xticks((np.arange(0, 10, 1)))
    plt.yticks((np.arange(0, 50, 7)))
    plt.savefig(path + r"case_b_h3.png", bbox_inches="tight", pad_inches=0)
    plt.imshow(attn_w_b[3], cmap='Blues', interpolation='nearest', origin='lower')
    plt.xticks((np.arange(0, 10, 1)))
    plt.yticks((np.arange(0, 50, 7)))
    plt.savefig(path + r"case_b_h4.png", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visual attn weights")
    # parser.add_argument("--topk",  type=int, metavar='N',
    #                     help="top k neighborhood viusal features")
    parser.add_argument("--save-dir", metavar="FP", default=None,
                       help="Path to the directory containing pickle files")
    # parser.add_argument('--gen-subset', default='test', metavar='SPLIT',
    #                    help='data subset to generate (train, valid, test)')
    args = parser.parse_args()

    # print(f'####{args.gen_subset}####')
    # print(f'R@{args.topk} Score:', end=" ")
    main(args)

