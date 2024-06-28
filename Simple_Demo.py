import torch
from model import BFGCN
import pickle

data_folder = './demo_data/demo_data.pickle'
plv_folder = './demo_data/demo_plv.pickle'

with open(data_folder, 'rb') as f:
    data = pickle.load(f)
with open(plv_folder, 'rb') as f:
    plv = pickle.load(f)

data = torch.Tensor(data).cuda()
plv = torch.Tensor(plv).cuda()
'''
    xdim: the shape of data
    kadj: k order of cheybnet
    nclass: the number of class
    num_out: the hidden dim of node feature
    att_hidden: the hidden dim of attention to fuse the three graph branches 
    att_plv_hidden: the hidden dim of attention to fuse the four bands plv matrix
    classifier_hidden: the hidden dim of classifier
    avgpool: the size of 2D average-pooling operation
    dropout: the rate of dropout
'''
model = BFGCN(xdim=[62, 5], kadj=2, nclass=3, num_out=16, att_hidden=16, att_plv_hidden=62, classifier_hidden=32, avgpool=2, dropout=0).cuda()
class_output, domain_output, emb1, com1, com2, emb2, emb, class_output1, Xcom, output_att, att = model(data, plv, alpha=0)
print('finish')
