import torch
import torch.nn as nn


class Decoder(nn.Module):
    # def __init__(self):
    #     super().__init__()
    def __init__(self, dropout_prob):
        super(Decoder, self).__init__()
        # self.train_W = nn.Parameter(train_W)
        # self.lin = nn.Linear(2261,215,bias=False)

        # self.w = torch.nn.Parameter(torch.randn(2568, 2568, dtype=torch.float64), requires_grad=True)
        # self.w = torch.nn.Parameter(torch.randn(218, 218, dtype=torch.float64), requires_grad=True)   # 1
        # self.w = torch.nn.Parameter(torch.randn(180, 180, dtype=torch.float64), requires_grad=True)  #2
        # self.w = torch.nn.Parameter(torch.randn(270, 270, dtype=torch.float64), requires_grad=True) # 3
        self.w = torch.nn.Parameter(torch.randn(154, 154, dtype=torch.float64), requires_grad=True)
        self.w = nn.init.xavier_normal_(self.w)

        self.dropout = nn.Dropout(dropout_prob)  # 添加 Dropout 层
        self.sig = nn.Sigmoid()

    def forward(self, H, drug_num, target_num,jz_fm,jz_fd):
        HR = H[0:drug_num]
        HD = H[drug_num:(drug_num + target_num)]
        hr_new = torch.cat((HR,jz_fm),dim=-1)
        hd_new = torch.cat((HD, jz_fd), dim=-1)
        supp1 = torch.mm(hr_new, self.w)

        supp1 = self.dropout(supp1)

        decoder = torch.mm(supp1, hd_new.transpose(0, 1))
        decoder = self.sig(decoder)

        return decoder