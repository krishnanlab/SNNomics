import torch
import torch.nn as nn
import torch.nn.functional as F


class SNN(nn.Module):
    def __init__(self, _in: int, task: str):
        super(SNN, self).__init__()
        self._in = _in
        self.task = task
    
        self.fc = nn.Sequential(
            nn.Linear(self._in, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 128),
        )

    def forward_once(self, x):
        output = self.fc(x)
        
        return output

    def forward(self, input1, input2, input3=None):
        if self.task == 'train':
            out1 = self.forward_once(input1)
            out2 = self.forward_once(input2)
            out3 = self.forward_once(input3)

            return out1, out2, out3
        
        elif self.task == 'predict':
            out1 = self.forward_once(input1)
            out2 = self.forward_once(input2)
            
            return out1, out2

