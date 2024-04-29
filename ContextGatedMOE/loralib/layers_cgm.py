import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


class ContextGatedMixing(nn.Module):
    '''
    If merging n LoRA layers, create output n*(lora_dim * lora_r)
    Then reshape output to match the lora shape of each layer

    Try first using one weight to merge loras at each layer
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hid_size = 4
      
        self.lin1 = nn.Linear(input_dim, hid_size)
        self.output = nn.Linear(hid_size, output_dim)
        
    def forward(self, x):
        # assert len(x.shape) == 3
        
        x = F.relu(self.lin1(x))
        x = self.output(x)
        
        # For transformer block inputs, we need to reduce the seq_len dim
        if len(x.shape) == 3:

            x = torch.mean(x, dim=1)
        
        return F.softmax(x)
        

class CGMLinear(nn.Linear):
    def __init__(
        self, 
        lora_A_weights: list,
        lora_B_weights: list,
        lora_alpha: int = 1, 
        **kwargs
    ):
        assert len(lora_A_weights) == len(lora_B_weights)

        # Create linear layer weights, this will be overridden when the base state dict is loaded
        in_features = lora_A_weights[0].size(1)
        out_features = lora_B_weights[0].size(0)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        r = lora_A_weights[0].size(0)
        self.scaling = lora_alpha / r

        self.lora_A = []
        self.lora_B = []
        for lora_A_weight, lora_B_weight in zip(lora_A_weights, lora_B_weights):
            self.lora_A.append(nn.Parameter(lora_A_weight))
            self.lora_B.append(nn.Parameter(lora_B_weight))

        self.context_gated_mixing = ContextGatedMixing(in_features, 2)

    def forward(self, x: torch.Tensor):        
        context_gated_coefs = self.context_gated_mixing(x)
        context_gated_coefs = torch.mean(context_gated_coefs, dim=0)

        a = context_gated_coefs[0]
        b = context_gated_coefs[1]

        # Create combined lora_A and lora_B
        lora_A_ensemble = a*self.lora_A[0] + b*self.lora_A[1]
        lora_B_ensemble = a*self.lora_B[0] + b*self.lora_B[1]

        #print(round(a.item(), 2), round(b.item(), 2), round(c.item(), 2))
        
        result = F.linear(x, self.weight, bias=self.bias)            
        result += (x @ lora_A_ensemble.transpose(0, 1) @ lora_B_ensemble.transpose(0, 1)) * self.scaling
        return result


