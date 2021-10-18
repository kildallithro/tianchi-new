import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from models import reduce_mem as rm


seed = 2021
data_path = './tcdata'

np.random.seed(seed)
random.seed(seed)


def get_user_items_list(all_click):
    train_user_ids = all_click['user_id'].unique()
    all_click = all_click[all_click['label'] == 1]

    all_2_tianchi = {}
    for u in train_user_ids:
        all_2_tianchi[u] = list(all_click[all_click['user_id'] == u]['click_article_id'])

    all_2_tianchi_list = []
    for user, items in tqdm(all_2_tianchi.items()):
        if len(items) == 0:
            continue
        all_2_tianchi_list.append([user, *items])

    return all_2_tianchi_list


def get_sample_2_tianchi(data_path):
    columns_name = ['user_id', 'click_article_id', 'label']
    train_click = pd.read_csv(data_path + '/train/train_all_click_sample.csv', sep=' ')
    train_click.columns = columns_name
    train_click = rm.reduce_mem(train_click)

    test_click = pd.read_csv(data_path + '/valid/valid_all_click_sample.csv', sep=' ')
    test_click.columns = columns_name
    test_click = rm.reduce_mem(test_click)

    train_2_tianchi_list = get_user_items_list(train_click)
    test_2_tianchi_list = get_user_items_list(test_click)

    return train_2_tianchi_list, test_2_tianchi_list


def get_df_2_tianchi(data_path):
    columns_name = ['user_id', 'click_article_id', 'label']
    train_click = pd.read_csv(data_path + '/train/train_all_click_df.csv', sep=' ')
    train_click.columns = columns_name
    train_click = rm.reduce_mem(train_click)

    test_click = pd.read_csv(data_path + '/valid/valid_all_click_df.csv', sep=' ')
    test_click.columns = columns_name
    test_click = rm.reduce_mem(test_click)

    train_2_tianchi_list = get_user_items_list(train_click)
    test_2_tianchi_list = get_user_items_list(test_click)

    return train_2_tianchi_list, test_2_tianchi_list


# train_2_tianchi_list_sample, test_2_tianchi_list_sample = get_sample_2_tianchi(data_path)
# with open("./tcdata/train/train_2_tianchi_sample.csv", "w") as f:
#     for i in range(len(train_2_tianchi_list_sample)):
#         for j in train_2_tianchi_list_sample[i]:
#             f.write(str(j) + " ")
#         f.write("\n")
#
# with open("./tcdata/test/test_2_tianchi_sample.csv", "w") as f:
#     for i in range(len(test_2_tianchi_list_sample)):
#         for j in test_2_tianchi_list_sample[i]:
#             f.write(str(j) + " ")
#         f.write("\n")

train_2_tianchi_list_df, test_2_tianchi_list_df = get_df_2_tianchi(data_path)
with open("./tcdata/train/train_2_tianchi_df.csv", "w") as f:
    for i in range(len(train_2_tianchi_list_df)):
        for j in train_2_tianchi_list_df[i]:
            f.write(str(j) + " ")
        f.write("\n")

with open("./tcdata/test/test_2_tianchi_df.csv", "w") as f:
    for i in range(len(test_2_tianchi_list_df)):
        for j in test_2_tianchi_list_df[i]:
            f.write(str(j) + " ")
        f.write("\n")
