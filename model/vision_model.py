
import torch
import torch.nn as nn




class Vision_model(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers 
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) 
        return x
            

    

        