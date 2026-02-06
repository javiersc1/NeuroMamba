import torch
import torch.nn as nn
import math
from torchinfo import summary
from transformers import PatchTSTConfig, PatchTSTModel, PatchTSTForRegression

class PatchTST(nn.Module):
    def __init__(self, n_layers = 3, n_heads=4, patch_length=16, patch_stride=8, d_model = 16, ffn_dim=128, dropout=0.2, score_amount=3, time_samples=570):
        super().__init__()
        config = PatchTSTConfig(
        num_input_channels=272,
        context_length=time_samples,
        patch_length=patch_length,
        patch_stride=patch_stride,
        mask_type=None,
        d_model=d_model,
        ffn_dim=ffn_dim,
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        norm_type="batchnorm",
        scaling=None,
        attention_dropout=dropout,
        ffn_dropout=dropout,
        num_targets=score_amount
        )
        self.backbone = PatchTSTForRegression(config)

    def forward(self, x):
        output = self.backbone(x)
        scores = output.regression_outputs
        #latents = output.hidden_states
        #x = x.mean(dim=1)
        #scores = self.regression(x)
        return scores


    
if __name__ == "__main__":
    
    model = PatchTST(n_layers = 3, n_heads=4, patch_length=16, patch_stride=8, d_model = 16, ffn_dim=128, dropout=0.2, score_amount=3).to("cuda")
    #x = torch.randn(4,570,272).to("cuda")
    #y = model(x)
    #print(y.shape)
    #print(scores.shape)
    #print(latents.shape)
    
    summary(model, (32,570,272))