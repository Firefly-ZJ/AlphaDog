#####     可视化  Visualization     #####
# 黑方先手, 1=黑, 2=白, 0=空
# 棋盘大小: 16*16

import pygame
import numpy as np
import torch

from _GomokuBoard import GomokuBoard
from _Player import PLAYER
from _HumanPlayer import HumanPl
from AlphaDog import AlphaDog
from BetaDog import BetaDog

### ----- Initialization and Constants ----- ###
FPS = 10 # Frames per second
pygame.init()
pygame.display.set_caption("Gomoku")

size = 16 ### Board size
step = 800 / (size+1)
clock = pygame.time.Clock()
font = pygame.font.SysFont('calibri', 60, True, True)

device = torch.device("cuda" if torch.cuda.is_available() else
                      "xpu" if torch.xpu.is_available() else "cpu")

### ----- Player Initialization ----- ###
AImodel = "./trained/model_v0.1.pth" if True else None

player1 = PLAYER(1) if True else HumanPl(1)
#player1 = AlphaDog(1, size, device, load_path=AImodel, num_simulations=100).playMode()
#player1 = BetaDog(1, num_simulations=2000)

#player2 = PLAYER(2) if False else HumanPl(2)
player2 = AlphaDog(2, size, device, load_path=AImodel, num_simulations=100).playMode()
#player2 = BetaDog(2, num_simulations=2000)

### ****************************************
class GomokuGame():
    """Black First. 1=Black, 2=White"""
    def __init__(self, pl1:PLAYER, pl2:PLAYER):
        self._Running:bool = True # Whether is running
        self._Board = GomokuBoard(size) # Gomoku Board
        self.screen = pygame.display.set_mode((1200,800))
        self.player1, self.player2 = pl1, pl2
        print("Player 1: ", pl1)
        print("Player 2: ", pl2)
        self.START() # Start a new game
    
    def START(self):
        """Start a new game"""
        self._Board.reset()
        self._NowPlayer = 1
        self.winner = None
        self.moves = [] # Moves history: [(y,x,side), ...]
        self.player1.StartNew()
        self.player2.StartNew()

        self.DrawBoard()
        self.ShowNowSide()
    
    def KILL(self):
        """Kill the game and exit"""
        pygame.quit()
        self._Running, self.winner = False, None
    
    def CheckKill(self, events) -> bool:
        """检查是否退出游戏"""
        for event in events:
            if event.type == pygame.QUIT:
                self.KILL()
                return True
            else: continue
        return False
    
    def GetNowPl(self) -> PLAYER:
        if self._NowPlayer == 1: return self.player1
        elif self._NowPlayer == 2: return self.player2
        else: return None
    
    def DrawBoard(self):
        """画棋盘和线"""
        pygame.draw.rect(self.screen, (238,173,14), pygame.Rect(0,0,800,800))
        pygame.draw.rect(self.screen, (205,190,112), pygame.Rect(800,0,400,800))

        color, lw = (0,0,0), 3
        for i in range(0, size):
            x_pos = step*(i+1)
            pygame.draw.line(self.screen, color, (x_pos,step), (x_pos,800-step), lw)
        for j in range(0, size):
            y_pos = step*(j+1)
            pygame.draw.line(self.screen, color, (step,y_pos), (800-step,y_pos), lw)
    
    def ShowNowSide(self):
        """显示当前方"""
        color = (0,0,0) if self._NowPlayer==1 else (255,255,255)
        center = (1000, 150)
        pygame.draw.circle(self.screen, color, center, radius=50)

    def DrawStone(self, y, x, side:int):
        """画一枚棋子：黑1，白2"""
        color = (0,0,0) if side==1 else (255,255,255)
        center = (step*(x+1), step*(y+1))
        pygame.draw.circle(self.screen, color, center, radius=step/3)
    
    def ShowWiningMove(self, y, x):
        """显示胜手"""
        color = (250,0,0)
        center = (step*(x+1), step*(y+1))
        pygame.draw.circle(self.screen, color, center, radius=step/6)

    def PlaceStone(self, y, x, side:int) -> bool:
        """放入一颗棋子：黑1，白2"""
        if self._Board.placeStone(y, x):
            self.moves.append((y, x, side))
            self.DrawStone(y, x, side)
            return True
        else: return False
    
    def ChangeSide(self) -> bool:
        """更换当前方"""
        self._NowPlayer = 1 if self._NowPlayer == 2 else 2
        self.ShowNowSide()
        return True
    
    def GetBoard(self) -> GomokuBoard:
        """获取当前board情况（返回副本）"""
        return self._Board.copy()
    
### ----- Main Loop ----- ###
print("Game Start")
Game = GomokuGame(player1, player2)
while Game._Running:
    if Game.winner != None:
        clock.tick(0.2) # 5s自动下一局
        Game.START()
    print()
    pygame.display.flip()
    if Game.CheckKill(pygame.event.get()): break

    ### 一局游戏
    while Game.winner is None:
        nowPl = Game.GetNowPl()
        clock.tick(FPS)

        events = pygame.event.get()
        if Game.CheckKill(events): break

        ### 玩家选择行动
        if nowPl.IsHuman():
            action = nowPl.ACT(events, step)
        else:
            with torch.no_grad():
                action = nowPl.ACT(Game.GetBoard())
        if action: Y, X = action
        else: continue

        ### 玩家落子成功
        if Game.PlaceStone(Y, X, Game._NowPlayer):
            if Game._Board.isWin(Y, X): ### 一方胜利，一局结束
                Game.winner = Game._NowPlayer
                nowPl.AddWin()
                Game.ShowWiningMove(Y, X)
                text_1 = "BALCK Win" if Game.winner==1 else "WHITE Win"
                Game.screen.blit(font.render(text_1,True,'black'),(850,250))
                print(text_1)
                print(nowPl)
                print(Game.moves)
            elif Game._Board.isDraw(): ### 和棋，一局结束
                Game.winner = False
                Game.screen.blit(font.render("Tied",True,'black'),(850,250))
                print("Tied")
                print(Game.moves)
            else: ### 本局继续，更换当前行动方
                Game.ChangeSide()
        else: continue

        pygame.display.flip()

print("Over")
