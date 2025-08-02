#####     GomokuBoard     #####
import numpy as np
import torch
from copy import deepcopy
from collections import deque

class GomokuBoard:
    """Gomoku Board: 0=Empty, 1=Black, 2=White"""
    def __init__(self, size:int=16):
        self.Size:int = size
        self.Board:np.ndarray = np.zeros((size, size), dtype=np.int8)
        self.PlayerNow = 1 # 1=Black, 2=White
        self.GameEnd, self.Winner = False, None
        self.moves:list = [] # Player's moves
        # History Steps Memory (5 steps) # new at right
        self.histSteps = 1
        self.History = deque([np.zeros((size, size), dtype=np.int8) for i in range(self.histSteps)],
                             maxlen=self.histSteps)
    
    def reset(self):
        """Start a new game"""
        self.Board.fill(0)
        self.PlayerNow = 1
        self.GameEnd, self.Winner = False, None
        self.moves = []
        for _ in range(self.histSteps):
            self.History.append(np.zeros((self.Size, self.Size), dtype=np.int8))
    
    def copy(self) -> "GomokuBoard":
        """Deepcopy current GomokuBoard"""
        return deepcopy(self)
    
    def __getitem__(self, pos) -> int:
        """0=Empty, 1=Black, 2=White, -1=Invalid"""
        y, x = pos
        if 0<=x<self.Size and 0<=y<self.Size:
            return self.Board[y,x]
        else: return -1
    
    def getBoard(self) -> np.ndarray:
        """Deepcopy current board array"""
        return deepcopy(self.Board)
    
    def getSize(self) -> int:
        return self.Size
    
    def isLegal(self, row, col) -> bool:
        """Whether the move is legal"""
        return self[row, col] == 0

    def placeStone(self, row, col) -> bool:
        """Put a stone at certain position"""
        if self.isLegal(row, col):
            self.Board[row, col] = self.PlayerNow
            self.addHistory(row, col)
            if self.isWin(row, col): # Player win, game end
                self.GameEnd = True
                self.Winner = self.PlayerNow
            elif self.isDraw(): # Draw, game end
                self.GameEnd = True
                self.Winner = False
            else: pass
            # switch player
            self.PlayerNow = 1 if self.PlayerNow == 2 else 2
            return True
        else:
            return False

    def addHistory(self, row, col):
        """Add the move to history memory (3 steps)"""
        position = np.zeros((self.Size, self.Size), dtype=np.int8)
        position[row, col] = 1
        self.History.append(position) # new at left
        self.moves.append((row, col, self.PlayerNow))
            
    def isWin(self, row, col, trial=False) -> bool:
        """Whether the player wins after the move"""
        # 1=Black, 2=White
        player = self[row,col] # completed action
        if trial and (player==0): player = self.PlayerNow # assumed action
        if not (player in (1,2)): return False

        directions = ((1,0), (0,1), (1,1), (1,-1))
        for dr, dc in directions:
            count = 1
            for delta in (-1, 1):
                r, c = row + delta*dr, col + delta*dc
                while self[r,c] == player:
                    count += 1
                    r += delta * dr
                    c += delta * dc
            if count >= 5: return True
        return False
    
    def isDraw(self) -> bool:
        """Whether the board is full and no one wins"""
        return np.all(self.Board != 0)

    def getState(self) -> np.ndarray:
        """Get current state [4,sz,sz]

        - 2 Layers: Player & Opponent
        - 1 Layer: Legal Position
        - 1 Layer: History Move
        """
        player = self.PlayerNow # 1=Black, 2=White
        opponent = 1 if player == 2 else 2

        playerPos = self.Board == player
        opponentPos = self.Board == opponent
        legalPos = self.Board == 0 # legal position
        #c = 1 if player == 1 else 0
        #color = np.full_like(playerPos, c)
        
        state = np.stack([playerPos, opponentPos, legalPos, *self.History], axis=0).astype(np.int8)
        return state
    
    def getStateAsT(self) -> torch.Tensor:
        """Get current state as Tensor [1,4,sz,sz]"""
        return torch.from_numpy(self.getState()).float().unsqueeze(0)

    def __str__(self):
        return "\n".join([" ".join(["X" if i==1 else "O" if i==2 else "." for i in row]) for row in self.Board])
    
if __name__ == "__main__":
    board = GomokuBoard()
    board.reset()
    print(board)
    print(board.getStateAsT().shape)
    print()

    board.placeStone(1,1)
    board.placeStone(1,2)
    print(board)
    print(board.getStateAsT())