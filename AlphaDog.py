#####     AI Player     #####
#####     AlphaDog     #####
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from _Player import PLAYER
from _GomokuNet import GomokuNet
from _GomokuBoard import *
from _MCTS import *
from _Dataset import Dataset

class AlphaDog(PLAYER):
    def __init__(self, side=1, size=16, device="cpu", load_path=None,
                 num_simulations=400, dataset_capacity=20000):
        """Gomoku AI Player: AlphaDog
        Args:
            side (int): player side (1 or 2)
            size (int): board size (default: 16)
            device (str, torch.device): device to use (default: "cpu")
            load_path (str): path to load pre-trained model weights
            num_simulations (int): number of MCTS simulations per move
            dataset_capacity (int): max size of the replay memory
        """
        super().__init__(side, name="AlphaDog", mode="AI")
        self.Size = size # board size
        self.dv = device if isinstance(device, torch.device) else torch.device(device)
        self.useMCTS = True ### Only active in Pygame playing!
        ### Net and MCTS
        self.Net = GomokuNet(size, half=True).to(self.dv)
        if load_path: self.Net.load_state_dict(torch.load(load_path, map_location="cpu", weights_only=True))
        self.MCTS = MCTS(self.Net, num_simulations, device=self.dv)
        self.Memory = Dataset(self.dv, capacity=dataset_capacity)
        ### Hyper-parameters
        self.tem = 1.01 # Temperature param (1/tau): visitCounts = visitCounts ** self.tem
        self.noise = True # Add Dirichlet Noise
        self.eps = 0.2 # Dir-Noise fraction: P(s,a) = (1-eps)*p(a) + eps*Dir(alpha)
    
    def StartNew(self):
        self.MCTS.reset()
    
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
    
    def playMode(self, useMCTS=True) -> 'AlphaDog':
        """Set to play mode (for Pygame playing)"""
        self.Net.eval()
        self.noise = False
        self.useMCTS = useMCTS
        return self
    
    def ACT(self, board:GomokuBoard, *args, **kwargs) -> tuple:
        """Pygame playing interface for action"""
        if self.useMCTS: # use MCTS search
            return self.TakeAction(board)
        else: # no search, only P-V net
            state = torch.from_numpy(board.getState()).half().unsqueeze(0).to(self.dv)
            policy, value = self.Net(state)
            policy = torch.exp(policy.squeeze())
            pos = torch.multinomial(policy, 1).item()
            return pos // self.Size, pos % self.Size
    
    def _SelfPlay(self):
        """Create self-play data and put into dataset"""
        board = GomokuBoard()
        self.StartNew()
        states, mctsProbs = [], []
        # Self-Play A Game
        while not board.GameEnd:
            boardState = torch.from_numpy(board.getState()).half().unsqueeze(0).to(self.dv)
            actionProbs = self.MCTS.search(board) # [sz*sz]
            pos = self.selectAction(actionProbs, moveCount=len(board.moves))
            if board.placeStone(pos // self.Size, pos % self.Size):
                states.append(boardState)
                mctsProbs.append(torch.from_numpy(actionProbs).half().to(self.dv))
        result = 1 if board.Winner == 1 else -1 if board.Winner==2 else 0
        # Update Replay Memory (one for each move)
        for state, mctsProb in zip(states, mctsProbs): # old first
            self.Memory.push(state, mctsProb, result)
            result = -result
    
    def CreateData(self, num_games:int):
        """Create self-play data"""
        self.Net.eval()
        with torch.no_grad():
            for i in range(num_games): self._SelfPlay()

    def SaveModel(self, path:str):
        """Save trained model (only weights)"""
        torch.save(self.Net.state_dict(), path)
        print("Model Saved")
    
    def TRAIN(self, num_epochs:int, num_games=24, lr=1e-3, batch_size=256):
        """Train the model
        Args:
            num_epochs (int): number of training epochs
            num_games (int): number of self-play games per epoch
            lr (float): init learning rate
            batch_size (int): mini-batch size
        """
        print("Training...")
        print(f"Epochs: {num_epochs}, Learning Rate: {lr},  Batch Size: {batch_size}")
        optimizer = torch.optim.AdamW(self.Net.parameters(), lr=lr, weight_decay=1e-3)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        for epoch in range(1, num_epochs+1):
            print(f"\nEpoch: {epoch} / {num_epochs}")
            optimizer.zero_grad()
            t_start = time.time()

            ### Self-Play
            self.CreateData(num_games)
            if not (len(self.Memory) > batch_size*2): continue
            else: print(f"Buffer: {len(self.Memory)}", end=",  ")

            ### Sample Randomly and Step
            iters = 0 # 迭代次数
            counter = 0 # 无效迭代计数器
            bestLoss = 100
            lossSum = 0.0
            self.Net.train()
            optimizer.zero_grad()
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
                #nn.utils.clip_grad_norm_(self.Net.parameters(), max_norm=1.0) # 梯度裁剪
                optimizer.step()
                # Check loss
                iters += 1
                lossSum += loss.item()
                if loss.item() >= bestLoss: counter += 1
                else: bestLoss, counter = loss.item(), 0
                optimizer.zero_grad()
            #scheduler.step()
            print(f"Iters: {iters},  Time: {(time.time()-t_start)/60:.1f} min")
            print(f"Loss = {lossSum/iters:.4f} ~ {bestLoss:.4f}  ({policyLoss.item():.4f} + {valueLoss.item():.4f})")
            if epoch % 5 == 0: self.Memory.clear() # clear old data
            if epoch % 2 == 0: self.SaveModel(rootPath+f"model_{epoch}.pth") # save model
        
        print("\nTraining Completed")

if __name__ == "__main__":
    rootPath = "./"
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "xpu" if torch.xpu.is_available() else "cpu")
    print(f"Device: {device}")

    player = AlphaDog(device=device, num_simulations=100)
    player.TRAIN(num_epochs=20, num_games=4, lr=5e-4, batch_size=64)