import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_


class MF_Naive(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, reg_u, reg_i, device='cuda'):
        super(MF_Naive, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.user_e = nn.Embedding(self.num_users, embedding_size)
        self.item_e = nn.Embedding(self.num_items, embedding_size)
        self.user_b = nn.Embedding(self.num_users, 1)
        self.item_b = nn.Embedding(self.num_items, 1)

        self.reg_u = reg_u
        self.reg_i = reg_i

        self.apply(self._init_weights)

        # self.loss = nn.MSELoss()
        self.loss = nn.BCELoss()  # 输入的元素要在 [0,1] 之间

        self.device = device

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)

    def forward(self, user, item):
        user_embedding = self.user_e(user)
        item_embedding = self.item_e(item)

        preds = (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        preds += self.user_b(user)
        preds += self.item_b(item)
        preds = torch.sigmoid(preds)

        return preds.squeeze()

    def calculate_loss(self, user_list, item_list, label_list):
        user_embedding = self.user_e(user_list)
        item_embedding = self.item_e(item_list)
        preds = self.forward(user_list, item_list)

        loss = self.loss(preds, label_list)
        # 正则项
        loss_reg = 0
        loss_reg = self.reg_u * torch.norm(user_embedding, 2)
        loss_reg += self.reg_i * torch.norm(item_embedding, 2)

        return loss + loss_reg

    def predict(self, user, item):
        return self.forward(user, item)

    def recommendation(self):
        user_list = torch.tensor(range(self.num_users)).to(self.device)
        item_list = torch.tensor(range(self.num_items)).to(self.device)
        user_embedding = self.user_e(user_list)
        item_embedding = self.item_e(item_list)

        preds_matrix = user_embedding @ item_embedding.T
        preds_matrix += self.user_b(user_list)
        preds_matrix += self.item_b(item_list).T
        preds_matrix = torch.sigmoid(preds_matrix)

        return preds_matrix

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def get_embedding(self):
        return self.user_e, self.item_e
