# coding:utf8
import warnings
import torch


class DefaultConfig(object):
    model = 'MF_Naive'
    is_sample = True
    is_eval_ips = False

    data_dir = './tcdata'

    # 采样数据
    train_data_sample = data_dir + '/train/train_all_click_sample.csv'
    val_data_sample = data_dir + '/valid/valid_all_click_sample.csv'
    # 全量数据
    train_data_df = data_dir + '/train/train_all_click_df.csv'
    val_data_df = data_dir + '/valid/valid_all_click_df.csv'

    test_data = data_dir + '/test/sample_submit.csv'

    reg_u = 0.001
    reg_i = 0.001

    metric = 'mrr'
    verbose = 5

    device = 'cuda'
    Ks = 20
    batch_size = 512
    embedding_size = 24

    max_epoch = 50
    lr = 0.001
    weight_decay = 1e-5

opt = DefaultConfig()
