# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn

from models.BaseModel import GeneralModel


class MF(GeneralModel):
    """
    Matrix Factorization (MF)
    r_ui = <p_u, q_i>
    """
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)

        self.emb_size = args.emb_size
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items

        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.user_emb = nn.Embedding(self.user_num, self.emb_size)
        self.item_emb = nn.Embedding(self.item_num, self.emb_size)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, feed_dict):
        """
        feed_dict:
            user_id: [batch_size]
            item_id: [batch_size, n_candidates]
        """
        user = feed_dict['user_id']
        items = feed_dict['item_id']

        u_emb = self.user_emb(user)                 # [B, d]
        i_emb = self.item_emb(items)                # [B, K, d]

        # 内积
        prediction = (u_emb[:, None, :] * i_emb).sum(dim=-1)

        return {
            'prediction': prediction.view(feed_dict['batch_size'], -1)
        }
