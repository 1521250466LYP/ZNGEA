import torch.nn as nn
import torch.nn.functional as F
from GCN import GCN
import torch


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]

class E_D(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(E_D, self).__init__()

        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)

        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nhid2)
        )

    def forward(self, x, fadj):

        emb = self.FGCN(x, fadj)


        emb = self.MLP(emb)

        [pi, disp, mean] = self.ZINB(emb)
        return emb, pi, disp, mean

