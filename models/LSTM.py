import torch
import torch.nn as nn
import math
from torchinfo import summary

class LSTM(nn.Module):
    def __init__(self, n_layers = 6, hidden_size= 272, bidirection = False, dropout=0.0, score_amount=3):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirection = bidirection
        self.D = 2 if self.bidirection else 1
        self.hidden_size = hidden_size
        self.backbone = nn.LSTM(272, self.hidden_size, num_layers=self.n_layers, batch_first=True, dropout=self.dropout, bidirectional=self.bidirection)
        self.regression = nn.Linear(self.D*self.hidden_size, score_amount)

    def forward(self, x):
        x = self.backbone(x)[0]
        x = x.mean(dim=1)
        scores = self.regression(x)
        return scores, x


    
if __name__ == "__main__":
    
    model = LSTM(n_layers = 8, hidden_size= 272, bidirection = True, dropout=0.0).to("cuda")
    # x = torch.randn(16,570,272).to("cuda")
    # scores, latents = model(x)
    # print(scores.shape)
    # print(latents.shape)
    
    summary(model, (16,570,272))