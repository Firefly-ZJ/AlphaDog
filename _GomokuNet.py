#####     AlphaDog Network    #####
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, chnls:int, knSize:int=5):
        """Residual Block (inChnl=outChnl)"""
        super().__init__()
        self.conv1 = nn.Conv2d(chnls, chnls, knSize, padding="same") # 5*5大卷积核，接归一化
        self.bn = nn.BatchNorm2d(chnls) # Batch Norm
        self.conv2 = nn.Conv2d(chnls, chnls*4, 1) # 1*1卷积核，接ReLU
        self.conv3 = nn.Conv2d(chnls*4, chnls, 1) # 1*1卷积核

    def forward(self, x): # f(x) = h(x) + x
        h = self.bn(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.conv3(h)
        return h + x

class GomokuNet(nn.Module):
    def __init__(self, size=16):
        """Gomoku neural network: State -> Policy, Value"""
        super().__init__()
        self.size = size
        self.convIn = nn.Sequential(nn.Conv2d(4, 96, kernel_size=9, padding="same"),
                                    nn.BatchNorm2d(96))
        self.res = nn.Sequential(ResBlock(96, knSize=5),
                                 ResBlock(96, knSize=5),
                                 ResBlock(96, knSize=5),
                                 ResBlock(96, knSize=5)) # ResBlocks * 4
        
        self.convP = nn.Sequential(nn.Conv2d(96, 2, kernel_size=1),
                                   nn.BatchNorm2d(2), nn.ReLU())
        self.fcP = nn.Linear(2*size*size, size*size)
        
        self.convV = nn.Sequential(nn.Conv2d(96, 1, kernel_size=1),
                                   nn.BatchNorm2d(1), nn.ReLU())
        self.fcV = nn.Sequential(nn.Linear(size*size, 128), nn.ReLU(),
                                 nn.Linear(128, 1))

    def forward(self, x): # [B, 4, sz, sz]
        x = self.convIn(x) # [B, 96, sz, sz]
        x = self.res(x) # [B, 96, sz, sz]

        policy = self.convP(x) # [B, 2, sz, sz]
        policy = policy.view(policy.size(0), -1) # [B, 2*sz*sz]
        policy = F.log_softmax(self.fcP(policy), dim=1) # [B, sz*sz]

        value = self.convV(x) # [B, 1, sz, sz]
        value = value.view(value.size(0), -1) # [B, 1*sz*sz]
        value = F.tanh(self.fcV(value)) # [B, 1]

        return policy, value

if __name__ == "__main__":
    from torchinfo import summary
    net = GomokuNet().eval()
    p, v = net(torch.randn(8, 4, 16, 16))
    print(p.size(), v.size(), "\n")
    print(summary(net, input_size=(128, 4, 16, 16)))