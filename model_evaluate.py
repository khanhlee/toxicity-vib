"""### Import library"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
from torch.utils.data import DataLoader,TensorDataset
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn import metrics
from sklearn.model_selection import train_test_split
import scipy.io as sio
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""### Pre-defined encoding methods"""

blosum62 = {
        'A': [4, -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0, -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        '-': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
    }

def normalization(dataset):
    min = dataset.min(axis=0)
    max = dataset.max(axis=0)
    dataset = (dataset - min) / (max - min)
    return dataset

def get_blosum62(seq):
    blosum_list = []
    for i in seq: 
        blosum_list.append(blosum62[i])
    blosum = np.array(blosum_list)
#     blosum = normalization(blosum)
    feature = np.zeros((1002,20))
    idx = blosum.shape[0]
    feature[0:idx,:] = blosum
    return feature

def make_tensor(path):
    data = pd.read_csv(path)
    sequences = data['sequence'].values
    labels = data['label'].values
    evolution = torch.zeros(len(sequences),1002,20)
    lengths = []
    for i in range(len(sequences)):
        lengths.append((len(sequences[i])))
        temp = get_blosum62(sequences[i])
        evolution[i,:,:] = torch.Tensor(temp)

    return evolution,torch.Tensor(lengths),torch.Tensor(labels)

def make_tensor2(S_train,y_train):
    sequences = S_train
    labels = y_train
    evolution = torch.zeros(len(sequences),1002,20)
    lengths = []
    for i in range(len(sequences)):
        lengths.append((len(sequences[i])))
        temp = get_blosum62(sequences[i])
        evolution[i,:,:] = torch.Tensor(temp)

    return evolution,torch.Tensor(lengths),torch.Tensor(labels)

"""### Neural Network Class"""

class dvib(nn.Module):
    def __init__(self,k,out_channels, hidden_size):
        super(dvib, self).__init__()
        
        self.conv = torch.nn.Conv2d(in_channels=1,
                            out_channels = out_channels,
                            kernel_size = (1,20),
                            stride=(1,1),
                            padding=(0,0),
                            )
        
        self.rnn = torch.nn.GRU(input_size = out_channels,  
                                hidden_size = hidden_size,
                                num_layers = 2,
                                bidirectional = True,
                                batch_first = True,
                                dropout = 0.2
                              )
        
        self.fc1 = nn.Linear(hidden_size*4, hidden_size*4)
#         self.fc2 = nn.Linear(1024,1024)
        self.hidden_dim = (hidden_size*4+578+k)//2
        self.enc_mean = nn.Linear(self.hidden_dim,k) # Concat
        self.enc_std = nn.Linear(self.hidden_dim,k) # Dense
        self.dec = nn.Linear(k, 2)        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size*4+578, (hidden_size*4+578+self.hidden_dim)//2),
            nn.ReLU(),
            nn.Linear((hidden_size*4+578+self.hidden_dim)//2, self.hidden_dim),
            nn.ReLU()
        )
        self.drop_layer = torch.nn.Dropout(0.5)
        # self.drop_layer2 = torch.nn.Dropout(0.1) # ++++
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_uniform_(self.enc_mean.weight)
        nn.init.constant_(self.enc_mean.bias, 0.0)
        nn.init.xavier_uniform_(self.enc_std.weight)
        nn.init.constant_(self.enc_std.bias, 0.0)
        nn.init.xavier_uniform_(self.dec.weight)
        nn.init.constant_(self.dec.bias, 0.0)
        
        
    def cnn_gru(self,x,lens):
        x = x.unsqueeze(1)
#         print(x.shape)
        x = self.conv(x)
#         print(x.shape)   
        x = torch.nn.ReLU()(x)
#         print(x.shape,type(x))
        x = x.squeeze(3)
#         x = x.view(x.size(0),-1)
        x = x.permute(0,2,1)
#         print(x.shape)
#         print(type(lens))
        gru_input = pack_padded_sequence(x,lens,batch_first=True)
        output, hidden = self.rnn(gru_input)
#         print(hidden.shape)
        output_all = torch.cat([hidden[-1],hidden[-2],hidden[-3],hidden[-4]],dim=1)
#         print("output_all.shape:",output_all.shape)    
        return output_all
    
        
    def forward(self, pssm, lengths, FEGS): 
        cnn_vectors = self.cnn_gru(pssm,lengths) # Tensor with shape torch.Size([100, 2048])
        feature_vec = torch.cat([cnn_vectors,FEGS],dim = 1) # Tensor with shape torch.Size([100, 2626])
        
        feature_vec = self.mlp(feature_vec)   # [100,1024]   
        
        enc_mean, enc_std = self.enc_mean(feature_vec), f.softplus(self.enc_std(feature_vec)-5) # Tensor with shape torch.Size([100, 1024])
        eps = torch.randn_like(enc_std) # Tensor with shape torch.Size([100, 1024])
        latent = enc_mean + enc_std*eps # Tensor with shape torch.Size([100, 1024])
        
        outputs = torch.sigmoid(self.dec(latent)) # Tensor with shape torch.Size([100, 2]) 
        # print(outputs.shape)

        return outputs,enc_mean, enc_std,latent

CE = nn.CrossEntropyLoss(reduction='sum')
betas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,1e-7]

def calc_loss(y_pred, labels, enc_mean, enc_std, beta=1e-3):
    """    
    y_pred : [batch_size,2]
    label : [batch_size,1]    
    enc_mean : [batch_size,z_dim]  
    enc_std: [batch_size,z_dim] 
    """   
    
    ce = CE(y_pred,labels)
    KL = 0.5 * torch.sum(enc_mean.pow(2) + enc_std.pow(2) - 2*enc_std.log() - 1)
    
    return (ce + beta * KL)/y_pred.shape[0]

"""### Protein pre-train"""

train_path = '/content/drive/My Drive/bioinformatics/Toxicity/dataset/protein_train1002.csv'
test_path = '/content/drive/My Drive/bioinformatics/Toxicity/dataset/protein_test1002.csv'

train_data = sio.loadmat('/content/drive/MyDrive/bioinformatics/Toxicity/dataset/protein_train.mat')
train_FEGS = torch.Tensor(normalization(train_data['FV']))

test_data = sio.loadmat('/content/drive/My Drive/bioinformatics/Toxicity/dataset/protein_test.mat')
test_FEGS = torch.Tensor(normalization(test_data['FV']))

train_pssm, train_len,train_label = make_tensor(train_path)
test_pssm, test_len,test_label = make_tensor(test_path)

train_data = DataLoader(TensorDataset(train_pssm, train_len,train_FEGS,train_label), batch_size=100, shuffle=True)
test_data = DataLoader(TensorDataset(test_pssm, test_len,test_FEGS, test_label), batch_size=100)

print("data done")

# Evaluate on protein testset
model.load_state_dict(torch.load('/content/drive/My Drive/bioinformatics/Toxicity/results/protein_best_model_weights_test200_VIB.pth'))
model.eval()
correct = 0
y_pre = []
y_test = []
with torch.no_grad():
    for batch_idx, (sequences, lengths,FEGS, labels) in enumerate(test_data):
        seq_lengths, perm_idx = lengths.sort(dim=0,descending=True)
        seq_tensor = sequences[perm_idx].to(device)
        FEGS_tensor = FEGS[perm_idx].to(device)
        label = labels[perm_idx].long().to(device)
#                     seq_lengths = seq_lengths.to(device)
        y_test.extend(label.cpu().detach().numpy())


        y_pred, end_means, enc_stds,latent = model(seq_tensor,seq_lengths,FEGS_tensor)
        y_pre.extend(y_pred.argmax(dim=1).cpu().detach().numpy())

#             pred = outputs.argmax(dim=1)
#             print(output.shape,label.shape)
        _, pred = torch.max(y_pred, 1) 

        correct += pred.eq(label).sum().item()
#             print(output,label.data)
#             correct += (output == label.data).sum()
    cm1 = metrics.confusion_matrix(y_test,y_pre)
    TP = cm1[1,1] # true positive 
    TN = cm1[0,0] # true negatives
    FP = cm1[0,1] # false positives
    FN = cm1[1,0] # false negatives
    SN = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('\nTest: Accuracy:{}/{} ({:.4f}%) Sensitivity:({:.2%}) f1:({:.2f}%) mcc:({:.2f}%)\n'.format(
        correct, len(test_data.dataset),
        100. * correct / len(test_data.dataset),
        SN,
        metrics.f1_score(y_test,y_pre),
        metrics.matthews_corrcoef(y_test,y_pre)
    ))

# Evaluate on peptide testset
model.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/peptide_best_model_weights_test200_VIB.pth'))
model.eval()
correct = 0
y_pre = []
y_test = []
with torch.no_grad():
    for batch_idx, (sequences, lengths,FEGS, labels) in enumerate(test_data):
        seq_lengths, perm_idx = lengths.sort(dim=0,descending=True)
        seq_tensor = sequences[perm_idx].to(device)
        FEGS_tensor = FEGS[perm_idx].to(device)
        label = labels[perm_idx].long().to(device)
#                     seq_lengths = seq_lengths.to(device)
        y_test.extend(label.cpu().detach().numpy())


        y_pred, end_means, enc_stds,latent = model(seq_tensor,seq_lengths,FEGS_tensor)
        y_pre.extend(y_pred.argmax(dim=1).cpu().detach().numpy())

#             pred = outputs.argmax(dim=1)
#             print(output.shape,label.shape)
        _, pred = torch.max(y_pred, 1) 

        correct += pred.eq(label).sum().item()
#             print(output,label.data)
#             correct += (output == label.data).sum()

    cm1 = metrics.confusion_matrix(y_test,y_pre)
    TP = cm1[1,1] # true positive 
    TN = cm1[0,0] # true negatives
    FP = cm1[0,1] # false positives
    FN = cm1[1,0] # false negatives
    SN = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('\nTest: Accuracy:{}/{} ({:.4f}%) Sensitivity:({:.2%}) f1:({:.2f}%) mcc:({:.2f}%)\n'.format(
        correct, len(test_data.dataset),
        100. * correct / len(test_data.dataset),
        SN,
        metrics.f1_score(y_test,y_pre),
        metrics.matthews_corrcoef(y_test,y_pre)
    ))