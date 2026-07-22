import torch
import torch.nn as nn

class LightCurveEncoder(nn.Module):
    def __init__(self,config:dict=None):
        super().__init__()
        if config and "light_curve_encoder" in config:
            self.output_dim = config["light_curve_encoder"].get("output_dim", 256)
        elif config and "output_dim" in config:
            self.output_dim = config["output_dim"]
        else:
            self.output_dim = 256
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        batch_size=x.shape[0]
        return torch.randn(batch_size,self.output_dim,dtype=torch.float32,device=x.device)
        