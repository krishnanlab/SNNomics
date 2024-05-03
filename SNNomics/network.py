import torch
import torch.nn as nn
import torch.nn.functional as F


class SNN(nn.Module):
    def __init__(self, _in: int):
        super(SNN, self).__init__()
        self._in = _in
    
        self.l0 = nn.Linear(self._in, 1000)
        self.l1 = nn.Linear(1000, 500)
        self.l2 = nn.Linear(500, 128)

    def forward_once(self, x):
        l0_out = F.relu(self.l0(x), inplace=True)
        l1_out = F.relu(self.l1(l0_out), inplace=True)
        l2_out = self.l2(l1_out)

        return l2_out

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        
        return out1, out2
