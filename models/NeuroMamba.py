import torch
import torch.nn as nn
import math
from torchinfo import summary
from mamba_ssm import Mamba
from mamba_ssm import Mamba2
    
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=64):
        super().__init__()
        self.dim = dim # C channels
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state)
    def forward(self, x):
        # print('x',x.shape)
        # B, L, C = x.shape
        # B = batch, L = sequence lenth time, C = channels or variables/ROIS
        x_mamba = self.norm(x)
        x_mamba = self.mamba(x_mamba)
        return x + x_mamba
    
class BiMambaBlock(nn.Module):
    def __init__(self, dim, d_state=64):
        super().__init__()
        self.dim = dim # C channels
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba1 = Mamba(d_model=dim, d_state=d_state, dt_max=1.0)
        self.mamba2 = Mamba(d_model=dim, d_state=d_state, dt_max=1.0)
    def forward(self, x):
        # print('x',x.shape)
        # B, L, C = x.shape
        # B = batch, L = sequence lenth time, C = channels or variables/ROIS
        x = self.norm(x)
        x1 = self.mamba1(x)
        x2 = torch.flip(self.mamba2(torch.flip(x, dims=(1,))), dims=(1,))
        x_mamba = self.norm2(x1 + x2)
        return x_mamba + x
    
class MambaPlusPlusBlock(nn.Module):
    def __init__(self, layer_idx, dim=272, d_state=64, dt_max=1.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ssm_forward = Mamba(d_model=dim, d_state=d_state, dt_max=dt_max)
        self.ssm_backward = Mamba(d_model=dim, d_state=d_state, dt_max=dt_max) 
        self.lambda_init = 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * self.layer_idx))
        self.lambda_q1 = nn.Parameter(torch.randn(dim))
        self.norm1 = nn.RMSNorm(dim, eps=1e-5)
        self.norm2 = nn.RMSNorm(dim, eps=1e-5)

    def forward(self, x):
        residual = x
        x = self.norm1(x) #self.norm(x.to(dtype=self.norm.weight.dtype))
        lambda_q1 = torch.sum(self.lambda_q1, dim=-1).float()
        lambda_full = torch.sigmoid(lambda_q1) + self.lambda_init

        y1 = self.ssm_forward(x)
        y2 = torch.flip(self.ssm_backward(torch.flip(x, dims=(1,))), dims=(1,))

        attn = y1 - lambda_full * y2
        attn = self.norm2(attn)
        x = attn * (1 - self.lambda_init)

        return residual + x
    
class MambaModel(nn.Module):
    def __init__(self, n_layers = 6, state_dim = 256, num_variables=272, score_amount=4):
        super().__init__()
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.dim = num_variables
        self.backbone = nn.Sequential(*[ nn.Sequential(MambaBlock(self.dim, d_state = self.state_dim)) for _ in range(self.n_layers)])     
        self.regression = nn.Linear(num_variables, score_amount)

    def forward(self, x):
        # x is 16, 570, 272
        x = self.backbone(x)
        x = x.mean(dim=1)
        scores = self.regression(x)
        return scores, x
    
class BiMambaModel(nn.Module):
    def __init__(self, n_layers = 6, state_dim = 256, num_variables=272, score_amount=4):
        super().__init__()
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.dim = num_variables
        self.backbone = nn.Sequential(*[ nn.Sequential(BiMambaBlock(self.dim, d_state = self.state_dim)) for _ in range(self.n_layers)])     
        self.regression = nn.Linear(num_variables, score_amount)

    def forward(self, x):
        # x is 16, 570, 272
        x = self.backbone(x)
        x = x.mean(dim=1)
        scores = self.regression(x)
        return scores, x

class NeuroMamba(nn.Module):
    def __init__(self, n_layers = 6, state_dim = 256, num_variables=272, score_amount=3, dt_max=1.0):
        super().__init__()
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.dim = num_variables
        self.backbone = nn.Sequential(*[ nn.Sequential(MambaPlusPlusBlock(layer_idx, dim=self.dim, d_state=self.state_dim, dt_max=dt_max)) for layer_idx in range(self.n_layers)])      
        self.regression = nn.Linear(num_variables, score_amount)

    def forward(self, x):
        # x is 16, 570, 272
        x = self.backbone(x)
        x = x.mean(dim=1)
        scores = self.regression(x)
        return scores, x
    
    def evaluate(self, x):
        # x is 16, 570, 272
        x = self.backbone(x)
        x2 = x.mean(dim=1)
        scores = self.regression(x2)
        return scores, x2, x
    
class NeuroMambaBCE(nn.Module):
    def __init__(self, n_layers = 12, state_dim = 32, num_variables=272):
        super().__init__()
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.dim = num_variables
        self.backbone = nn.Sequential(*[ nn.Sequential(MambaPlusPlusBlock(layer_idx, dim=self.dim, d_state=self.state_dim)) for layer_idx in range(self.n_layers)])      
        #self.norm = nn.LayerNorm(num_variables)
        self.mlp = nn.Linear(num_variables, num_variables)
        self.predict = nn.Linear(num_variables +1, 1)
    
    def forward(self, x, y):
        # x is 16, 570, 272
        x = self.backbone(x)
        x2 = x.mean(dim=1)
        x2 = self.mlp(x2)
        #x2 = self.norm(x2)
        x2 = torch.relu(x2)
        #x2 = self.norm(x2)
        features = torch.cat([x2, y], dim=-1)
        logits = self.predict(features)
        return logits, x2  
    
    def evaluate(self, x, y):
        # x is 16, 570, 272
        x = self.backbone(x)
        x2 = x.mean(dim=1)
        x2 = self.mlp(x2)
        #x2 = self.norm(x2)
        x2 = torch.relu(x2)
        #x2 = self.norm(x2)
        features = torch.cat([x2, y], dim=-1)
        logits = self.predict(features)
        m = nn.Sigmoid()
        probs = m(logits)
        return probs, x2
    
if __name__ == "__main__":
    model = NeuroMamba2(n_layers = 12, state_dim = 32, num_variables=272).to("cuda")
    x = torch.randn(4, 570, 272).to("cuda")
    y, latents = model(x)
    print(x.shape)
    print(y.shape)
    print(latents.shape)
    #summary(model, (16,570,272))