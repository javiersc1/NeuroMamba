import torch
import torch.nn as nn
from torchinfo import summary
 
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
 
    def forward(self, x):
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
 
 
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0, score_amount=3):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=1, padding='same', dropout=dropout)]
 
        self.network = nn.Sequential(*layers)
        self.regression = nn.Linear(num_channels[-1], score_amount)
 
    def forward(self, x):
        x = self.network(x.permute(0, 2, 1)).permute(0, 2, 1).mean(dim=1)
        scores = self.regression(x)
        return scores, x
    
if __name__ == "__main__":
    
    model = TemporalConvNet(num_inputs=272, num_channels=[272, 368, 512], kernel_size=3, dropout=0.0, score_amount=3).to("cuda")
    #x = torch.randn(32,570,272).to("cuda")
    #y = model(x)
    #print(y.shape)
    # scores, latents = model(x)
    # print(scores.shape)
    # print(latents.shape)
    
    summary(model, (16,570,272))
