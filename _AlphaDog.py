#####     AI Player     #####
#####     AlphaDog     #####
import numpy as np
import time, random
import _Player
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from _GomokuBoard import *
from _MCTS import *

#####     Dataset     #####
class Dataset():
    def __init__(self, device, capacity:int=100000):
        """Data buffer for training"""
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.dv = device

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear the data buffer"""
        self.buffer.clear()
    
    def push(self, state:torch.Tensor, mctsProb:torch.Tensor, result:int):
        """Put data into dataset (DA: rotation & flip)"""
        size = state.shape[-1]
        state = state.squeeze(0) # [1, C, sz, sz] -> [C, sz, sz]
        mctsProb = mctsProb.reshape(size, size) # [sz*sz] -> [sz, sz] -> [sz*sz]
        for k in range(4): # Rotation * 4
            S_ = torch.rot90(state, k, dims=(-2,-1))
            P_ = torch.rot90(mctsProb, k, dims=(-2,-1)).reshape(-1)
            self.buffer.append((S_, P_, result))
        state, mctsProb = torch.flip(state, dims=(-1,)), torch.flip(mctsProb, dims=(-1,))
        for k in range(4): # Rotation * 4
            S_ = torch.rot90(state, k, dims=(-2,-1))
            P_ = torch.rot90(mctsProb, k, dims=(-2,-1)).reshape(-1)
            self.buffer.append((S_, P_, result))

    def sample(self, batchSize:int):
        """Sample a batch randomly"""
        batch = random.sample(self.buffer, batchSize)
        states, mctsProbs, results = zip(*batch)
        states = torch.stack([s for s in states]).to(self.dv) # [C, sz, sz] -> [B, C, sz, sz]
        mctsProbs = torch.stack([p for p in mctsProbs]).to(self.dv) # [sz*sz] -> [B, sz*sz]
        results = torch.tensor(results, dtype=torch.float32).unsqueeze(1).to(self.dv) # num -> [B, 1]
        return states, mctsProbs, results

#####     Nural Network     #####
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

    def forward(self, x): # [B, 5, sz, sz]
        x = self.convIn(x) # [B, 128, sz, sz]
        x = self.res(x) # [B, 128, sz, sz]

        policy = self.convP(x) # [B, 2, sz, sz]
        policy = policy.view(policy.size(0), -1) # [B, 2*sz*sz]
        policy = F.log_softmax(self.fcP(policy), dim=1) # [B, sz*sz]

        value = self.convV(x) # [B, 1, sz, sz]
        value = value.view(value.size(0), -1) # [B, 1*sz*sz]
        value = F.tanh(self.fcV(value)) # [B, 1]

        return policy, value

#####     AlphaDog     #####
class AlphaDog(_Player.PLAYER):
    def __init__(self, side=1, size=16, modelPath=None, device=None):
        """AlphaDog: A Gomoku AI Inspired By AlphaZero"""
        super().__init__(side, name="AlphaDog", mode="AI")
        self.Size = size # board size
        self.dv = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ### Net and MCTS
        self.Net = GomokuNet(size=16).to(self.dv)
        if modelPath: self.Net.load_state_dict(torch.load(modelPath))
        self.MCTS = MCTS(self.Net, num_simulations=400, device=self.dv)
        self.Memory = Dataset(self.dv, capacity=20000)
        ### Hyper-parameters
        self.tem = 1.01 # Temperature param (1/tau): visitCounts = visitCounts ** self.tem
        self.noise = True # Add Dirichlet Noise
        self.eps = 0.2 # Dir-Noise fraction: P(s,a) = (1-eps)*p(a) + eps*Dir(alpha)
    
    def StartNew(self):
        self.MCTS.reset()
        
    def evalState(self, state:torch.Tensor):
        """State -> Policy, Value"""
        policy, value = self.Net.forward(state)
        return policy, value
    
    def selectAction(self, probs:np.ndarray, moveCount:int=0) -> int:
        """Select an action based on probability (with Tau and Dir-noise)"""
        ### Temperature
        addTem = 20 # start to add temperature
        if moveCount <= addTem: pass
        else:
            temperature = self.tem ** (moveCount-addTem)
            if temperature <= 30: # t=1.05 -> step=70
                probs = probs ** temperature
                probs = probs / np.sum(probs)
            else:
                pos = np.argmax(probs)
                probs = np.zeros_like(probs)
                probs[pos] = 1.0
        ### Dirichlet Noise
        if self.noise:
            probs = (1-self.eps) * probs + self.eps * np.random.dirichlet([0.1]*len(probs))
        action = np.random.choice(len(probs), p=probs)
        return action
    
    def TakeAction(self, board:GomokuBoard) -> tuple:
        """Take an action: Board -> (r,c)"""
        actionProbs = self.MCTS.search(board)
        pos = self.selectAction(actionProbs, moveCount=len(board.moves))
        return pos // self.Size, pos % self.Size
    
    def playMode(self) -> 'AlphaDog':
        self.Net.eval()
        self.noise = False
        return self
    
    def ACT(self, board:GomokuBoard, useMCTS:bool=True) -> tuple:
        ### Pygame 接口
        if useMCTS: # use MCTS search
            return self.TakeAction(board)
        else: # no search, only P-V net
            policy, _ = self.evalState(board.getStateAsT().to(self.dv))
            policy = torch.exp(policy.squeeze())
            pos = torch.multinomial(policy, 1).item()
            return pos // self.Size, pos % self.Size
    
    def _SelfPlay(self):
        """Create self-play data and put into dataset"""
        board = GomokuBoard()
        self.StartNew()
        states, mctsProbs = [], []
        # Self-Play A Game
        with torch.no_grad():
            while not board.GameEnd:
                boardState = board.getStateAsT().to(self.dv)
                actionProbs = self.MCTS.search(board) # [sz*sz]
                pos = self.selectAction(actionProbs, moveCount=len(board.moves))
                if board.placeStone(pos // self.Size, pos % self.Size):
                    states.append(boardState)
                    mctsProbs.append(torch.from_numpy(actionProbs).float().to(self.dv))
        result = 1 if board.Winner == 1 else -1 if board.Winner==2 else 0
        # Update Replay Memory (one for each move)
        for state, mctsProb in zip(states, mctsProbs): # old first
            self.Memory.push(state, mctsProb, result)
            result = -result
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def SaveModel(self, path:str):
        """Save trained model (only weights)"""
        torch.save(self.Net.state_dict(), path)
        print("Model Saved")
    
    def TRAIN(self, lr=1e-3, batch_size=256):
        """Train the model
        Args:
            lr: init learning rate
            batch_size: mini-batch size
        """
        num_epochs = 40
        self.Net.train()
        print(f"Learning Rate: {lr},  Batch Size: {batch_size}")
        optimizer = torch.optim.AdamW(self.Net.parameters(), lr=lr, weight_decay=1e-3)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        for epoch in range(1, num_epochs+1):
            print(f"Epoch: {epoch}")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            t_start = time.time()

            ### Self-Play
            for _ in range(12):
                self._SelfPlay()
            if not (len(self.Memory) > batch_size): continue
            else: print(f"Buffer:{len(self.Memory)}", end=",  ")

            ### Sample Randomly and Step
            iters = 0 # 迭代次数
            counter = 0 # 无效迭代计数器
            bestLoss = 100
            lossSum = 0.0
            while (counter <= 10) and (iters <= 100):
                optimizer.zero_grad()
                states_batch, mctsProbs_batch, rewards_batch = self.Memory.sample(batch_size)
                policy, value = self.evalState(states_batch)
                # LOSS: Policy-交叉熵损失 + Value-均方差损失 (+ WeightDecay)
                policyLoss = torch.mean(torch.sum(-mctsProbs_batch * (policy.to(self.dv)), dim=1))
                valueLoss = F.mse_loss(value.to(self.dv), rewards_batch)
                loss = policyLoss + valueLoss
                # Backward and Update
                loss.backward()
                nn.utils.clip_grad_norm_(self.Net.parameters(), max_norm=1.0) # 梯度裁剪
                optimizer.step()
                # Check loss
                iters += 1
                lossSum += loss.item()
                if loss.item() >= bestLoss: counter += 1
                else: bestLoss, counter = loss.item(), 0
                del states_batch, mctsProbs_batch, rewards_batch, policy, value
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            #scheduler.step()
            print(f"Iters:{iters},  Time:{(time.time()-t_start)/60:.1f}m")
            print(f"Loss: {lossSum/iters:.4f} ~ {bestLoss:.4f},  {policyLoss.item():.4f} + {valueLoss.item():.4f}")
            if epoch % 5 == 0: self.Memory.clear() # clear old data
            if epoch % 10 == 0: self.SaveModel(rootPath+f"model_{epoch}.pth") # save model
            print()
        
        print("Training Completed")

if __name__ == "__main__":
    rootPath = "./"
    player = AlphaDog(modelPath=None)
    print(f"{player}  |  {player.dv}")

    player.TRAIN()
