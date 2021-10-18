import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import numpy as np
import argparse
import random
import torch
import copy
import math
import gc
import pickle
from datetime import datetime
from operator import itemgetter
from collections import defaultdict
import collections

from metrics import AUC
from utils import MF_DATA, evaluate_model, evaluate_model_matrix
from config import opt
import models

import warnings
warnings.filterwarnings('ignore')

seed_num = 2021
print("seed_num:", seed_num)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed_num)
data_path = './tcdata'


# train for MF_Naive and MF_IPS
def train():
    print('train begin')

    if opt.is_sample:
        train_all_data = MF_DATA(opt.train_data_sample)
        val_data = MF_DATA(opt.val_data_sample)
    else:
        train_all_data = MF_DATA(opt.train_data_df)
        val_data = MF_DATA(opt.val_data_df)

    train_data = copy.deepcopy(train_all_data)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)

    model = getattr(models, opt.model)(train_data.users_num,
                                       train_data.items_num,
                                       opt.embedding_size,
                                       opt.reg_u, opt.reg_i,
                                       opt.device)  # getattr(x, 'y') is equivalent to x.y.

    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)

    best_mse = 10000000.
    best_mae = 10000000.
    best_auc = 0
    best_ndcg = 0
    best_mrr = 0
    best_iter = 0

    model.train()
    for epoch in range(opt.max_epoch):
        t1 = time()
        for i, data in tqdm(enumerate(train_dataloader)):
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(), item.long(),
                                        label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time()

        ndcg, mrr = evaluate_model_matrix(model, val_data, opt)

        if opt.metric == 'ndcg':
            if ndcg > best_ndcg:
                best_ndcg, best_iter = ndcg, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-ndcg-model.pth")
        elif opt.metric == 'mrr':
            if mrr > best_mrr:
                best_mrr, best_iter = mrr, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mrr-model.pth")

        if epoch % opt.verbose == 0:
            print('Epoch %d [%.1f s]:', epoch, t2 - t1)
            print('Train Loss = ', loss.item())
            print('Val NDCG = %.4f, MRR = %.4f [%.1f s]' % (ndcg, mrr, time() - t2))
            print("------------------------------------------")

    print("train end\nBest Epoch %d:  NDCG = %.4f, MRR = %.4f" %
          (best_iter, best_ndcg, best_mrr))

    best_model = getattr(models, opt.model)(train_all_data.users_num,
                                            train_all_data.items_num,
                                            opt.embedding_size,
                                            opt.reg_u, opt.reg_i,
                                            opt.device)

    best_model.to(opt.device)

    if opt.metric == 'ndcg':
        best_model.load_state_dict(torch.load("./checkpoint/ci-ndcg-model.pth"))
    elif opt.metric == 'mrr':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mrr-model.pth"))

    print("\n========================= best model =========================")
    ndcg, mrr = evaluate_model_matrix(best_model, train_data, opt)
    print('Train NDCG = %.4f, MRR = %.4f' % (ndcg, mrr))
    ndcg, mrr = evaluate_model_matrix(best_model, val_data, opt)
    print('Val NDCG = %.4f, MRR = %.4f' % (ndcg, mrr))
    print("===============================================================\n")

    return best_model


# gengerate submit file
def generate_submit(model):
    # 获取测试集
    test_data = np.loadtxt(opt.test_data, dtype=int, skiprows=1, delimiter=',')
    test_user = np.unique(test_data[:, 0])

    # 从所有召回数据中将测试集中的用户选出来
    train_recall = model.recommendation().to(opt.device)
    train_recall = train_recall.detach().cpu().numpy()
    test_recall = train_recall[train_recall[:, 0] == test_user, :]

    # 给定用户，选出前 topK 的文章 id
    ranking = np.argsort(test_recall)[:, ::-1]
    ranking = ranking[:, :opt.Ks]

    submit = np.hstack((test_data[:, 0], ranking))
    np.savetxt("submit.csv", submit, header=True, fmt='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--model', default='MF_Naive')  # MF_Naive MF_IPS CausE 换模型的话，改这里的参数
    parser.add_argument('--is_sample', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--metric',
                        default='mrr',
                        choices=["ndcg", "mrr"])

    args = parser.parse_args()
    opt.model = args.model
    opt.is_sample = args.is_sample
    opt.batch_size = args.batch_size
    opt.max_epoch = args.epoch
    opt.lr = args.lr
    opt.metric = args.metric

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))
    print('model is', opt.model)

    best_model = train()
    # generate_submit(best_model)

    print('end')
