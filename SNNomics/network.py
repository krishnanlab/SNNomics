import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer():
    def __init__(self, _in: int, _out: int):
        super().__init__()

        self.layer = nn.Linear(_in, _out)
    
    def forward(self, x):
        
        z = self.layer(x)
        net_out = F.relu(z, inplace=True) 

        return out


class SNN():
    def __init__(self, _in: list, _out: list):
        super().__init__()
    
        self.l0 = Layer(_in[0], _out[0])
        self.l1 = Layer(_in[1], _out[1])
        self.l2 = Layer(_in[2], _out[2])

    def forward_once(self, x)
        l0_out = self.l0(x)
        l1_out = self.l1(l0_out)
        l2_out = self.l2(l1_out)

        return l2_out

    def forward(self, input1, input2)
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        
        return out1, out2


