import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import wandb

from methods.meta_template import MetaTemplate

class CanNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, reduction_ratio=6, temperature=0.025, scale_cls=7, num_classes=7195):
        super(CanNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.m = self.feat_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(1, self.num_classes)
        self.fusion_conv = nn.Conv1d(self.feat_dim, 1, kernel_size=1)
        self.bn = nn.BatchNorm1d(int(self.feat_dim / reduction_ratio))
        self.w1 = nn.Linear(self.m, int(self.m / reduction_ratio))
        self.activation = nn.ReLU()
        self.w2 = nn.Linear(int(self.m / reduction_ratio), self.m)
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = temperature
        self.scale_cls = scale_cls
        self.weight_factor = 0.5
        self.cosine_distance = nn.CosineSimilarity(dim=2, eps=1e-6)

    def set_forward(self, x, is_feature=False):
        # Compute the prototypes (support) and queries (embeddings) for each datapoint.
        z_support, z_query = self.parse_feature(x, is_feature)
            
        # Compute the prototype.
        z_support = z_support.contiguous().view(self.n_way, self.n_support, -1)
        z_proto = z_support.mean(dim=1)
        
        # Format the queries for the similarity computation.
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        z_proto_attention, z_query_attention = self.cross_attention_module(z_proto, z_query)
        
        # ftest is used for the global classification loss, the second loss
        ftest = z_query_attention
        
        # z_proto_attention: torch.Size([5, 75, 64]) ie. [n_way, n_query, feat_dim]
        # z_query_attention: torch.Size([5, 75, 64]) ie. [n_way, n_query, feat_dim]
        z_proto_attention_mean = z_proto_attention.mean(dim=1) # torch.Size([5, 64])
        z_query_attention_mean = z_query_attention.mean(dim=0) # torch.Size([75, 64])

        # Compute similarity score based on the euclidean distance between prototypes and queries.
        #scores = -euclidean_dist(z_query, z_proto)
        scores = -euclidean_dist(z_query_attention_mean, z_proto_attention_mean)

        # use cosine similarity instead of euclidean distance for the scores
        #score_cosine = self.cosine_distance(z_query_attention_mean.unsqueeze(1), z_proto_attention_mean.unsqueeze(0))
        #scores = score_cosine

        return scores, ftest

    def set_forward_loss(self, x, y_true_query):
        # Compute the similarity scores between the prototypes and the queries.
        scores, ftest = self.set_forward(x)
        
        # Create the category labels for the queries.
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query).to(self.device)

        # Compute the knn loss (base protonet loss)
        l1 = self.loss_fn(scores, y_query)

        # Compute the global classification loss

        def one_hot(labels_train):
            """
            Turn the labels_train to one-hot encoding.
            Args:
                labels_train: [batch_size, num_train_examples]
            Return:
                labels_train_1hot: [batch_size, num_train_examples, K]
            """
            labels_train = labels_train.cpu()
            nKnovel = 1 + labels_train.max()
            labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
            labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
            labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
            return labels_train_1hot

        y_query_one_hot = one_hot(y_query).cuda()
        # ftest is of shape (5, 75, 64), change it to (1, 75, 64, 5) to be able to do matmul
        ftest = ftest.unsqueeze(0) # torch.Size([1, 5, 75, 64])
        ftest = ftest.transpose(2, 3) # torch.Size([1, 5, 64, 75])
        ftest = ftest.transpose(1, 3) # torch.Size([1, 75, 64, 5])
        
        # this matmul is incorrect should be ftest: (1, 75, 64, 5) and y_query_one_hot: (1, 75, 5, 1)
        y_query_one_hot = y_query_one_hot.unsqueeze(0) # torch.Size([1, 5, 75, 5])
        y_query_one_hot = y_query_one_hot.unsqueeze(3) # torch.Size([1, 5, 75, 5, 1])
        ftest = torch.matmul(ftest, y_query_one_hot) # torch.Size([1, 75, 64, 1])
        ftest = ftest.view(-1, self.m) # torch.Size([75, 64])

        ftest = ftest.unsqueeze(2) # torch.Size([75, 64, 1])

        ytest = self.linear(ftest) # torch.Size([75, 64, 59])
        ytest = ytest.transpose(2, 1) # torch.Size([75, 59, 64])
        y_true_query = y_true_query.reshape(-1) #torch.Size([75])

        # special loss used in the paper
        criterion = CrossEntropyLoss()

        # compute the global classification loss
        l2 = criterion(ytest, y_true_query)
        loss = self.weight_factor * l1 + l2

        return loss, l1, l2
    def fusion_layer(self, z):
        """
        Generates cross attention map A
        :param R: [n_dim,n_dim]
        :return: A  [n_dim]
        """

        GAP = torch.mean(z, dim=-2)

        w = self.w2(self.activation(self.w1(GAP)))


        fusion = z * w.unsqueeze(2)

        conv = torch.mean(fusion,dim=-1)

        A = self.softmax(conv/self.temperature)

        return A

    def cross_attention_module(self, z_support, z_query):

        def correlation_layer(z_support, z_query): 
            """
            Takes 1 support embedding and 1 query embedding and returns correlation map
            ie. P and Q in the paper. P is [P1, P2, ..., Pn] where n is the dimension of the embeddings, same for Q.
            :param z_support: [n_dim] ie. P
            :param z_query: [n_dim] ie. Q
            :return: correlation_map: [n_dim, n_dim]. Note: we use R^q = correlation_map and R^p = correlation_map.T 
            """

            # compute cosine similarity between support and query embeddings
            P = F.normalize(z_support, p=2, dim=-1, eps=1e-12)
            Q = F.normalize(z_query, p=2, dim=-1, eps=1e-12)
            P = z_support
            Q = z_query

            correlation_map = torch.einsum("ij,kl->ikjl",P,Q)  # dim: [n_dim, n_dim]

            return correlation_map

        P_k = z_support
        Q_b = z_query

        # compute correlation map
        R_p = correlation_layer(P_k, Q_b)
        R_q = R_p.transpose(2, 3)

        # compute fusion layer
        A_p = self.fusion_layer(R_p)
        A_q = self.fusion_layer(R_q)
        P_bk = P_k.unsqueeze(1) * (1 + A_p)
        Q_bk = Q_b.unsqueeze(0) * (1 + A_q)

        return P_bk, Q_bk

    def train_loop(self, epoch, train_loader, optimizer):
        """
        Same training loop as base, but added labels for the global classification loss
        """
        print_freq = 10

        avg_loss = 0
        avg_l1 = 0
        avg_l2 = 0
        for i, (x, y) in enumerate(train_loader):
            y_all = y.cuda()

            # y true query is the global labels for the query set
            y_true_query = y_all[:, self.n_support:]

            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else: 
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
            optimizer.zero_grad()
            loss, l1, l2 = self.set_forward_loss(x, y_true_query)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            avg_l1 = avg_l1 + l1.item()
            avg_l2 = avg_l2 + l2.item()


            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({"loss/train": avg_loss / float(i + 1)})
                wandb.log({"loss/train_loss1": avg_l1 / float(i + 1)})
                wandb.log({"loss/train_loss2": avg_l2 / float(i + 1)})
    
    def correct(self, x):
        # Compute the predictions scores.
        scores, _ = self.set_forward(x)

        # Compute the top1 elements.
        topk_scores, topk_labels = scores.data.topk(k=1, dim=1, largest=True, sorted=True)

        # Detach the variables (transforming to numpy also detach the tensor)
        topk_ind = topk_labels.cpu().numpy()

        # Create the category labels for the queries, this is unique for the few shot learning setup
        y_query = np.repeat(range(self.n_way), self.n_query)

        #>>> np.repeat(range(10), 2)
        #array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])

        # Compute number of elements that are correctly classified.
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)
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

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets): # inputs: torch.Size([75, 59, 64]), targets: torch.Size([75])
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        # inputs: torch.Size([75, 59, 64])
        log_probs = self.logsoftmax(inputs)

        # below = problematic line
        # torch zeros (75, 59)
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()
        loss = (- targets * log_probs).mean(0).sum() 
        return loss / inputs.size(2)

