import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Match(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(Match, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.ReLU(),
            # nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
            nn.Linear(self.output_size, self.output_size),
            nn.ReLU()
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, feat):
        return self.net(feat)


class Prob(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Prob, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.ReLU(),
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            nn.Linear(32, 2)
            # nn.Softmax()
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, feat):
        return self.net(feat)


class FrameByFrame(nn.Module):
    def __init__(self, Vinput_size=512, Ainput_size=128, output_size=64, layers_num=5, dropout=0):
        super(FrameByFrame, self).__init__()
        self.Vinput_size = Vinput_size
        self.Ainput_size = Ainput_size
        self.layers_num = layers_num
        self.output_size = output_size
        self.AFeatRNN = nn.LSTM(
            self.Ainput_size, self.output_size, self.layers_num)
        self.Amatching = Match(
            self.output_size, self.output_size, self.output_size, dropout)
        self.Vmatching = Match(
            self.Vinput_size, self.output_size, self.output_size, dropout)
        self.Prob = Prob(2*self.output_size, self.output_size, dropout)

    def forward(self, Vfeat, Afeat):
        # Afeat = [64*3, 128, 10]
        # Vfeat = [64*3, 512, 10]
        h_0 = Variable(torch.zeros(self.layers_num, Afeat.size(
            0), self.output_size), requires_grad=False)
        c_0 = Variable(torch.zeros(self.layers_num, Afeat.size(
            0), self.output_size), requires_grad=False)
        if Vfeat.is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        outAfeat, _ = self.AFeatRNN((Afeat/100.0).permute(2, 0, 1), (h_0, c_0))
        count = 0
        for i in range(10):
            # outAfeat: [10, 64*3, 64]
            # outAfeat[i,:,:]: [64*3, 64]
            Afeats = self.Amatching(outAfeat[i, :, :])
            # Vfeat: [64*3, 512, 10]
            # Vfeat[:,:,i]: [64*3, 512]
            Vfeats = self.Vmatching(Vfeat[:, :, i])
            feat = torch.cat((Afeats, Vfeats), dim=1)
            if i == 0:
                prob = self.Prob(feat)
                count += 1
            else:
                p = self.Prob(feat)
                avg = prob / count
                # if torch.max(p) > 2*prob/count:
                if False:
                    prob += 2*p
                    count += 2
                else:
                    prob += p
                    count += 1
        # prob = prob/10
        prob = prob / count
        return prob
