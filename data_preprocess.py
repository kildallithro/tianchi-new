import numpy as np
import pandas as pd
import copy
import random
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from models import reduce_mem as rm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


seed = 2021
data_path = './tcdata'

np.random.seed(seed)
random.seed(seed)


# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该将测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path, offline=True):
    if offline:
        all_click = pd.read_csv(data_path + '/train/train_click_log.csv')
        all_click = rm.reduce_mem(all_click)
    else:
        trn_click = pd.read_csv(data_path + '/train/train_click_log.csv')
        tst_click = pd.read_csv(data_path + '/train/testA_click_log.csv')

        all_click = trn_click.append(tst_click)
        all_click = rm.reduce_mem(all_click)

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id']))
    all_click['label'] = 1
    all_click = all_click.loc[:, ['user_id', 'click_article_id', 'label']]

    # 在 DataFrame 数据结构下求 user 列和 item 列去重后的个数
    all_click.sort_values(['user_id', 'click_article_id'], ascending=True)
    users_num = all_click['user_id'].unique().shape[0]
    items_num = all_click['click_article_id'].unique().shape[0]
    print('users_num: {}, items_num: {}'.format(users_num, items_num))

    # 每个用户采样 10 个负样本(label 列为 0)
    all_click = np.array(all_click)
    df_item_ids = list(np.unique(all_click[:, 1]))
    for idx, u in tqdm(enumerate(range(250000))):
        t1 = time()

        item_list = list(all_click[all_click[:, 0] == u, 1])
        tmp = copy.copy(df_item_ids)
        for i in item_list:
            tmp.remove(i)
        sample_item_ids = np.random.choice(tmp, size=4, replace=False)  # 负采样

        # 这个循环几乎占了一个 epoch 所有用时
        for si in sample_item_ids:
            row = [u, si, 0]
            all_click = np.row_stack((all_click, row))

        if u % 1000 == 0:
            print("Epoch %d: %.4f [%.2f]" % (u, u/250000, time()-t1))

    train, valid = train_test_split(all_click, test_size=0.25, random_state=42)
    return train, valid


# debug 模式：从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器内存的限制，可以采样用户数）
    """
    trn_click = pd.read_csv(data_path + '/train/train_click_log.csv')
    tst_click = pd.read_csv(data_path + '/test/testA_click_log.csv')

    all_click = trn_click.append(tst_click)
    all_click = rm.reduce_mem(all_click)
    all_user_ids = all_click.user_id.unique()

    # 随机采样：np.random.choice()
    # 对50万用户进行随机采样，而不是对110万条交互数据
    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]  # 对行的筛选

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id']))
    all_click['label'] = 1
    all_click = all_click.loc[:, ['user_id', 'click_article_id', 'label']]

    # 对第 1、2 列进行硬编码
    le1 = preprocessing.LabelEncoder()
    encoder1 = le1.fit(all_click['user_id'])
    all_click['user_id'] = encoder1.transform(all_click['user_id'])

    le2 = preprocessing.LabelEncoder()
    encoder2 = le2.fit(all_click['click_article_id'])
    all_click['click_article_id'] = encoder2.transform(all_click['click_article_id'])

    # 在 DataFrame 数据结构下求 user 列和 item 列去重后的个数
    # all_click.sort_values(['user_id', 'click_article_id'], ascending=True)
    users_num = all_click['user_id'].unique().shape[0]
    items_num = all_click['click_article_id'].unique().shape[0]
    print('users_num: {}, items_num: {}'.format(users_num, items_num))

    # 每个用户采样 10 个负样本(label 列为 0)
    all_click = np.array(all_click)
    df_item_ids = list(np.unique(all_click[:, 1]))
    for idx, u in tqdm(enumerate(range(sample_nums))):
        t1 = time()

        item_list = list(all_click[all_click[:, 0] == u, 1])
        tmp = copy.deepcopy(df_item_ids)
        for i in item_list:
            tmp.remove(i)
        sample_item_ids = np.random.choice(tmp, size=10, replace=False)

        for si in sample_item_ids:
            row = [u, si, 0]
            all_click = np.row_stack((all_click, row))

        if u % 100 == 0:
            print("Epoch %d: %.4f [%.2f]" % (u, u/sample_nums, time()-t1))

    train, valid = train_test_split(all_click, test_size=0.25, random_state=42)
    return train, valid

# 读取采样数据
# train_all_click_sample, valid_all_click_sample = get_all_click_sample(data_path, 10000)
# np.savetxt("./tcdata/train/train_all_click_sample.csv", train_all_click_sample, fmt=('%d', '%d', '%d'))
# np.savetxt("./tcdata/valid/valid_all_click_sample.csv", valid_all_click_sample, fmt=('%d', '%d', '%d'))

# 读取全量数据
train_all_click_df, valid_all_click_df = get_all_click_df(data_path, offline=False)
np.savetxt("./tcdata/train/train_all_click_df.csv", train_all_click_df, fmt=('%d', '%d', '%d'))
np.savetxt("./tcdata/valid/valid_all_click_df.csv", valid_all_click_df, fmt=('%d', '%d', '%d'))