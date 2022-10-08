import pickle
import numpy as np
import torch
import argparse, logging

def main(args):
    # save_dir = '/home/think/checkpoints/multi30k/en-de/inverseKD_res_m_l2_1enc'
    syn_f = open(args.save_dir + f'/{args.gen_subset}SynLocal_v.pickle', 'rb')
    org_f = open(args.save_dir + f'/{args.gen_subset}OrgLocal_v.pickle', 'rb')

    # load numpy from .pkl file
    syn_out = []
    org_out = []
    while True:
        try:
            syn_i = pickle.load(syn_f)
            org_i = pickle.load(org_f)
            syn_out.append(syn_i)
            org_out.append(org_i)
        except EOFError:
            break

    # numpy array
    syn_arr = np.stack(syn_out, axis=0)
    org_arr = np.stack(org_out, axis=0)

    # torch tensor
    syn_mat = torch.from_numpy(syn_arr)
    org_mat = torch.from_numpy(org_arr)

    # consine similarity
    cos_sim = torch.mm(syn_mat, org_mat.T) / (torch.norm(syn_mat) * torch.norm(org_mat))
    _, indices = cos_sim.topk(k=args.topk, dim=1, largest=True, sorted=True) # sorted by column
    batch = cos_sim.shape[0]
    indexs = torch.arange(0, batch).view(-1, 1).repeat(1, args.topk)
    mask = (indices == indexs)
    rk_score = mask.sum(dim=1).sum(dim=0) / batch
    print(rk_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visual grounding")
    parser.add_argument("--topk",  type=int, metavar='N',
                        help="top k neighborhood viusal features")
    parser.add_argument("--save-dir", metavar="FP", default=None,
                       help="Path to the directory containing pickle files")
    parser.add_argument('--gen-subset', default='test', metavar='SPLIT',
                       help='data subset to generate (train, valid, test)')
    args = parser.parse_args()

    print(f'####{args.gen_subset}####')
    print(f'R@{args.topk} Score:', end=" ")
    main(args)

