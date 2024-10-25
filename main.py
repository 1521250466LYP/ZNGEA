from sklearn.model_selection import KFold
from getsample import *
from NMF import *
import warnings
from data_processing import *
from Encoder_Decoder import *
from torch import *
import torch.optim as optim
from bilineardecoder import Decoder
from sklearn import metrics
from similarity_fusion import *
import numpy as np
from ZINB import *

warnings.filterwarnings("ignore")


# parameter
n_splits = 5
epochs_ZINB = [60]
fold = 0
result = np.zeros((1, 7), dtype=np.float32)
nhid1 = 512
nhid2 = 64
fdim = 2478
epoch_Decoder = 10000
dropout = 0.2
lr_ZINB = 0.0001
weight_decay = 5e-4
decoder_dropout=0.4
decoder_lr =0.0001

def caculate_metric(pre_score,real_score):
    y_pre=pre_score
    y_true=real_score

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)

    y_score = [0 if j < 0.5 else 1 for j in y_pre]

    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)

    metric_result = [auc, aupr, acc, f1, recall, precision]
    print("One epoch metric： ")
    print_met(metric_result)


    return metric_result


def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'f1_score ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'precision ：%.4f \n' % (list[5]))


# ZINB训练
def train():
    model.train()
    optimizer.zero_grad()
    emb, pi, disp, mean = model(features_tensor,train_adj_tensor)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features_tensor, mean, mean=True)
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    zinb_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss



def train_final(ml,fea,sam,label,jz_m,jz_d):
    loss1 = nn.BCELoss()
    label_new = (torch.from_numpy(label)).cuda()
    ml.train()
    optimizer.zero_grad()
    md_score = ml(fea,2262,216,jz_m,jz_d)
    aa = md_score[tuple(sam.T)]
    los = loss1(md_score[tuple(sam.T)],(label_new.squeeze(dim=-1)).to(torch.float64))
    print(los)
    los.backward()
    optimizer.step()
    train_score = aa.cpu().detach().numpy()
    result = caculate_metric(train_score, label)
    return md_score








association = pd.read_csv("M_D.csv", index_col=0).to_numpy()

samples = get_all_samples(association)


k1 = 226
k2 = 21
m_fusion_sim, d_fusion_sim = get_fusion_sim(k1, k2)
#
# m_fusion_sim = pd.read_csv("m_fusion_sim.csv", index_col=0, dtype=np.float32).to_numpy()
# d_fusion_sim = pd.read_csv("d_fusion_sim.csv", index_col=0, dtype=np.float32).to_numpy()

kf = KFold(n_splits=n_splits, shuffle=True)


D = 90

a = 0.0001
NMF_mfeature, NMF_dfeature = get_low_feature(D, 0.01, a, association)

m_number = association.shape[0]
d_number = association.shape[1]
all_number = m_number+d_number

features = np.zeros((all_number, all_number))
adj = np.zeros((all_number, all_number))

features[:m_number, :d_number] = association
features[m_number:, d_number:] = association.T

features[:m_number, d_number:] = m_fusion_sim
features[m_number:, :d_number] = d_fusion_sim

adj[:m_number, m_number:] = association
adj[m_number:, :m_number] = association.T


diagonal_matrix = np.eye(all_number)
adj_A = adj+diagonal_matrix

sum_valid_result = [0, 0, 0, 0, 0, 0]

for train_index, val_index in kf.split(samples):
    fold += 1
    train_samples = samples[train_index, :]
    val_samples = samples[val_index, :]

    train_adj = adj_A.copy()
    for i in val_samples:

        train_adj[i[0], i[1]+m_number] = 0
        train_adj[m_number + i[1], i[0]] = 0

    norm_train_adj = norm_adj(train_adj)

    train_adj_tensor = torch.FloatTensor(norm_train_adj)
    features_tensor = torch.FloatTensor(features)

    model = E_D(nfeat=fdim,
                         nhid1=nhid1,
                         nhid2=nhid2,
                         dropout=dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        features_tensor = features_tensor.to(device)
        train_adj_tensor = train_adj_tensor.to(device)


    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_ZINB, weight_decay=weight_decay)
    for epoch in range(0, epochs_ZINB[0]):
        emb, mean, zinb_loss = train()
        print(f"{epoch}epoch loss = {zinb_loss}")





    model_final = Decoder(decoder_dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        feat = (torch.from_numpy(emb)).to(device)


    model_final.to(device)
    optimizer = optim.Adam(model_final.parameters(), lr=decoder_lr, weight_decay=0.0005)
    JZ_m = (torch.from_numpy(NMF_mfeature)).cuda()
    JZ_d = (torch.from_numpy(NMF_dfeature)).cuda()


    train_sam = (train_samples[:, 0:2]).astype(np.int64)
    train_la = (train_samples[:, 2:]).astype(np.float32)
    valid_sam = (val_samples[:, 0:2]).astype(np.int64)
    valid_la = (val_samples[:, 2:]).astype(np.float32)
    for epoch in range(0, epoch_Decoder):
        md_s = train_final(model_final,feat,train_sam,train_la,JZ_m,JZ_d)
        print(epoch)

    model_final.eval()
    with torch.no_grad():
        val_score =model_final(feat,2262,216,JZ_m,JZ_d)
        pre = val_score[tuple(valid_sam.T)]
        pre = pre.cpu().detach().numpy()
        print("验证集")
        valid_result = caculate_metric(pre, valid_la)

        sum_valid_result = [x + y for x, y in zip(sum_valid_result, valid_result)]
print("平均结果")
sum_valid_result = [x / 5 for x in sum_valid_result]
print_met(sum_valid_result)












