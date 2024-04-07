import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import ReverseLayerF


def normalize_A(A):
    A = F.relu(A)
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt(d + 1e-10)
    D = torch.diag_embed(d)
    L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(A.shape[1]).cuda())
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support


class GraphConvolution(nn.Module):

    def __init__(self, num_in, num_out, bias=False):

        super(GraphConvolution, self).__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out).cuda())
        nn.init.kaiming_normal_(self.weight, )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_out).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out, dropout):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        self.dp = nn.Dropout(dropout)
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[1], num_out))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class Attentionadj(nn.Module):
    def __init__(self, in_size, hidden_size=62):
        super(Attentionadj, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class BFGCN(nn.Module):
    def __init__(self, nclass, xdim, kadj, num_out, att_hidden, att_plv_hidden, classifier_hidden ,avgpool, dropout):
        super(BFGCN, self).__init__()
        self.feature = SFGCN(xdim, kadj, num_out, att_hidden, att_plv_hidden, dropout, avgpool)
        self.class_classifier = nn.Sequential(
            nn.Linear(int(xdim[0]*num_out/(avgpool**2)), classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.output = nn.Linear(classifier_hidden, nclass)
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(int(xdim[0]*num_out/(avgpool**2)), 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, fadj, alpha):
        feature, att, emb1, com1, com2, emb2, emb, Xcom, output_att = self.feature(input_data, fadj)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output1 = self.class_classifier(feature)
        class_output = self.output(class_output1)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output, emb1, com1, com2, emb2, emb, class_output1, Xcom, output_att, att


class SFGCN(nn.Module):
    def __init__(self, xdim, kadj, num_out, att_hidden, att_plv_hidden, dropout, avgpool):
        super(SFGCN, self).__init__()

        self.SGCN1 = Chebynet(xdim, kadj, num_out, dropout)
        self.SGCN2 = Chebynet(xdim, kadj, num_out, dropout)
        self.CGCN = Chebynet(xdim, kadj, num_out, dropout)
        self.BN1 = nn.BatchNorm1d(xdim[1])
        self.attention = Attention(num_out, att_hidden)
        self.attentionadj = Attentionadj(xdim[0], att_plv_hidden)
        self.bn = nn.BatchNorm1d(xdim[0])
        self.mp = nn.AvgPool2d(avgpool)
        self.A = nn.Parameter(torch.FloatTensor(xdim[0], xdim[0]).cuda())
        self.A = nn.init.kaiming_normal_(self.A, )

    def forward(self, x, fadj):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        # weighted sum 4 band matrix to 1 matrix
        fadj = fadj.permute(0, 3, 1, 2)
        fadj, att = self.attentionadj(fadj)
        # normalize and construct Laplace matrix
        fadj = normalize_A(fadj)
        sadj = normalize_A(self.A + torch.eye(self.A.shape[0]).cuda())
        # chebynet gcn
        emb1 = self.SGCN1(x, sadj)
        com1 = self.CGCN(x, sadj)
        com2 = self.CGCN(x, fadj)
        emb2 = self.SGCN2(x, fadj)
        Xcom = (com1 + com2) / 2
        # attention mechanism
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        output, att = self.attention(emb)
        output_att = output
        output = F.relu(self.bn(output))
        # global graph feature
        output = self.mp(output)
        output = output.reshape(output.shape[0], -1)
        return output, att, emb1, com1, com2, emb2, emb, Xcom, output_att