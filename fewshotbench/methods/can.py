# This code is modified from https://github.com/blue-blue272/fewshot-CAN

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate


class CanNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(CanNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.linear = nn.Linear(self.feat_dim, n_way)

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)


        for i in range(num_classes):
            z_proto, z_query = cross_attention_module(z_proto, z_query)
            

        # keep euclidean distance for now
        # TODO: change this
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        # cls scores
        cls_scores = self.linear(z_query)
        return scores, cls_scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores, cls_scores = self.set_forward(x)

        l1 = self.loss_fn(scores, y_query )
        l2 = self.loss_fn(cls_scores, y_query )
        loss = (l1 + l2) / 2
        

        return loss



def cross_attention_module(z_support, z_query):
    """
    TODO: do this operation for all pairs at once instead of looping, look at base code
    Takes 1 support embedding and 1 query embedding and returns cross-attentioned embeddings
    :param z_support: [n_dim]
    :param z_query: [n_dim]
    """


    def correlation_layer(z_support, z_query): 
        """
        Takes 1 support embedding and 1 query embedding and returns correlation map
        ie. P and Q in the paper. P is [P1, P2, ..., Pn] where n is the dimension of the embeddings, same for Q.
        :param z_support: [n_dim] ie. P
        :param z_query: [n_dim] ie. Q
        :return: correlation_map: [n_dim, n_dim]. Note: we use R^q = correlation_map and R^p = correlation_map.T 
        """

        # compute cosine similarity between support and query embeddings
        P = z_support / torch.linalg.norm(z_support, dim=1, ord=2, keepdim=True)
        Q = z_query / torch.linalg.norm(z_query, dim=1, ord=2, keepdim=True)
        correlation_map = P @ Q.T # dim: [n_dim, n_dim]

        return correlation_map
        

    def fusion_layer(z_support, z_query):
        """
        Generates cross attention maps A^p and A^q
        :param z_support: [n_dim] ie. R^p
        :param z_query: [n_dim] ie. R^q
        :return: A^p and A^q, both [n_dim]
        """
        pass

    P_k = z_support
    Q_b = z_query

    # compute correlation map
    R_p = correlation_layer(P_k, Q_b)
    R_q = R_p.T

    # compute fusion layer
    A_p, A_q = fusion_layer(R_p, R_q)


    A_p = A_p * P_k
    P_bk = A_p + P_k

    A_q = A_q * Q_b
    Q_bk = A_q + Q_b

    return P_bk, Q_bk




    
def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
