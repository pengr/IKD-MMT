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
    _, indices = cos_sim.topk(k=args.topk, dim=-1, largest=True, sorted=True) # sorted by column
    print("line 653 (case A) : a young boy holding a small artsy soccer ball wearing white frills on his wrists .") # a ballerina in blue twirls .
    print(indices[417])

    # visualize a clustering figure via Matplotlib toolkit
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import KMeans

    data = np.random.rand(100, 2)
    #生成一个随机数据，样本大小为100, 特征数为2（这里因为要画二维图，所以就将特征设为2，至于三维怎么画？
    #后续看有没有机会研究，当然你也可以试着降维到2维画图也行）
    plt.figure(figsize=(7, 7.5))

    estimator = KMeans(n_clusters=7)  #构造聚类器，构造一个聚类数为3的聚类器
    estimator.fit(data)  #聚类
    label_pred = estimator.labels_ #获取聚类标签

    centroids = estimator.cluster_centers_ #获取聚类中心
    inertia = estimator.inertia_ # 获取聚类准则的总和
    mark = ['or', 'ob', 'og', 'ok', 'oy', 'om', 'oc', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    #这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推

    color = 0
    j = 0
    for i in label_pred:
        plt.plot([data[j:j+1, 0]], [data[j:j+1, 1]], mark[i], markersize=7)
        j +=1
    # legend = ax.legend(loc='upper right')

    # hidden the Coordinate scale
    plt.xticks([])
    plt.yticks([])
    plt.savefig("cluster.jpg")
    plt.show()

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

