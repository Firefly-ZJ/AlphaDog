#####     Dataset     #####
import numpy as np
import random
from collections import deque
import torch

#####     Dataset     #####
class Dataset():
    """Data buffer for training"""
    def __init__(self, device="cpu", capacity:int=100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.dv = device if isinstance(device, torch.device) else torch.device(device)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear the data buffer"""
        self.buffer.clear()
    
    def push(self, state:torch.Tensor, mcts_prob:torch.Tensor, result:int):
        """Put data into dataset (DA: rotation & flip)"""
        size = state.shape[-1]
        state = state.squeeze(0) # [1, C, sz, sz] -> [C, sz, sz]
        mcts_prob = mcts_prob.reshape(size, size) # [sz*sz] -> [sz, sz] -> [sz*sz]
        for k in range(4): # Rotation * 4
            S_ = torch.rot90(state, k, dims=(-2,-1))
            P_ = torch.rot90(mcts_prob, k, dims=(-2,-1)).reshape(-1)
            self.buffer.append((S_, P_, result))
        state, mcts_prob = torch.flip(state, dims=(-1,)), torch.flip(mcts_prob, dims=(-1,))
        for k in range(4): # Rotation * 4
            S_ = torch.rot90(state, k, dims=(-2,-1))
            P_ = torch.rot90(mcts_prob, k, dims=(-2,-1)).reshape(-1)
            self.buffer.append((S_, P_, result))

    def sample(self, batch_size:int):
        """Sample a batch randomly, then load to device"""
        batch = random.sample(self.buffer, batch_size)
        states, mctsProbs, results = zip(*batch)
        states = torch.stack([s for s in states]).to(self.dv) # [C, sz, sz] -> [B, C, sz, sz]
        mctsProbs = torch.stack([p for p in mctsProbs]).to(self.dv) # [sz*sz] -> [B, sz*sz]
        results = torch.tensor(results, dtype=torch.float32).unsqueeze(1).to(self.dv) # num -> [B, 1]
        return states, mctsProbs, results

if __name__ == "__main__":
    pass