#####     AI Player     #####
#####     AlphaDog (parallel training)     #####
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import multiprocessing
multiprocessing.set_sharing_strategy('file_system') ###

from _GomokuNet import GomokuNet
from _GomokuBoard import *
from _MCTS import *
from _Dataset import Dataset
from AlphaDog import AlphaDog

class AlphaDogPara(AlphaDog):
    """Parallel version of AlphaDog to accelerate training"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _SelfPlay(self) -> list[tuple]:
        """Create self-play data
        Returns:
            [(state, mctsProb, result), ...]
        """
        board = GomokuBoard()
        self.StartNew()
        states, mctsProbs = [], []
        # Self-Play A Game
        with torch.no_grad():
            while not board.GameEnd:
                boardState = board.getStateAsT() # [1, C, sz, sz]
                actionProbs = self.MCTS.search(board) # [sz*sz]
                pos = self.selectAction(actionProbs, moveCount=len(board.moves))
                if board.placeStone(pos // self.Size, pos % self.Size):
                    states.append(boardState)
                    mctsProbs.append(torch.from_numpy(actionProbs).float())
        result = 1 if board.Winner==1 else -1 if board.Winner==2 else 0
        # Return Data
        data = []
        for state, mctsProb in zip(states, mctsProbs): # old first
            data.append((state.cpu(), mctsProb.cpu(), result))
            result = -result
        if not (result <= 0): raise RuntimeError("Wrong game result!")
        del states, mctsProbs
        return data

    def CreateData(self, numGames:int, numPara:int=4):
        """Create self-play data (support cuda multi-processing)
        Args:
            numGames (int): num of self-play data to be created
            numPara (int): num of parallel workers (if > 1)
        """
        self.Net.eval()
        with torch.no_grad():
            if numPara > 1: ### Parallel Self-Play
                modelDict = self.Net.state_dict()
                args = [(modelDict, self.dv) for _ in range(numGames)]
                with multiprocessing.Pool(numPara) as pool:
                    allData = pool.starmap(_selfplayWorker, args)
            else: ### Sequential Self-Play (when numPara<=1)
                allData = []
                for _ in range(numGames): allData.append(self._SelfPlay())
        for data in allData: ### Update Replay Memory
            for state, mctsProb, result in data:
                self.Memory.push(state, mctsProb, result)
        del allData
    
    def TRAIN(self, num_epochs:int, num_games=64, lr=1e-3, batch_size=256):
        """Train the model
        Args:
            num_epochs (int): number of training epochs
            num_games (int): number of self-play games per epoch
            lr (float): init learning rate
            batch_size (int): mini-batch size
        """
        print("Training...")
        print(f"Epochs: {num_epochs}, Learning Rate: {lr},  Batch Size: {batch_size}\n")
        optimizer = torch.optim.AdamW(self.Net.parameters(), lr=lr, weight_decay=1e-3)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        
        for epoch in range(1, num_epochs+1):
            print(f"Epoch: {epoch}/{num_epochs}")
            optimizer.zero_grad()
            t_start = time.time()
            
            ### Self-Play
            self.CreateData(num_games, numPara=4)
            if not (len(self.Memory) > batch_size*2): continue
            else: print(f"Buffer: {len(self.Memory)}", end=",  ")

            ### Sample and Step
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
            print(f"Loss={lossSum/iters:.4f} ~ {bestLoss:.4f},  {policyLoss.item():.4f} + {valueLoss.item():.4f}")
            if epoch % 5 == 0: self.Memory.clear() # clear old data
            if epoch % 10 == 0: self.SaveModel(rootPath+f"model_{epoch}.pth") # save model
            print()
        
        print("Training Completed")

def _selfplayWorker(modelDict, device:torch.device):
    """Self-Play Worker (for parallel processing)"""
    #torch.cuda.set_per_process_memory_fraction(0.2)
    player = AlphaDogPara(device=device) # new player, same net
    player.Net.load_state_dict(modelDict)
    player.Net.eval()
    return player._SelfPlay()

###
if __name__ == "__main__":
    rootPath = "./"
    multiprocessing.set_start_method("spawn") ###
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "xpu" if torch.xpu.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available(): print(torch.cuda.get_device_properties(0))

    player = AlphaDogPara(device=device, load_path=None)
    player.TRAIN(num_epochs=32, num_games=64)