#####     Search Player (Beta Dog)    #####
import numpy as np
#import multiprocessing as mp
from _Player import PLAYER
from _GomokuBoard import GomokuBoard
from _GomokuBoard_c import *

#####     Node     #####
class Node():
    def __init__(self, board:CmprsBoard, priorProb:float=1.0, isWin:bool=False):
        self.isValid = True # 有效节点
        self.board = board # 该节点的棋盘状态
        self.children:dict[int,Node] = dict() # 子节点: K-Action, V-ChildNode
        self.visitCount = 0  # 该节点的访问次数
        self.priorProb = priorProb # 该节点的先验概率
        self.valueSum = 0 # 该节点的价值和
        self.isWin = isWin # 是否为胜利节点

    def haveChildren(self) -> bool:
        """Whether is already expanded (has children)"""
        return len(self.children) > 0

    def getValue(self) -> float:
        """Mean value of the node: valueSum/visitCount"""
        if self.visitCount == 0: return 0
        else: return self.valueSum / self.visitCount
    
    def getActionProbs(self, tem=1.0) -> np.ndarray:
        """action probs of children: based on visitCounts"""
        visitCounts = np.array([child.visitCount for child in self.children.values()])
        if tem != 1.0: visitCounts = visitCounts ** tem
        return visitCounts / np.sum(visitCounts) # [sz*sz]

class InvalidNode(Node):
    """Invalid Move Node (no children)"""
    __slots__ = ("visitCount", "isValid")
    def __init__(self):
        self.isValid = False # 无效节点
        self.visitCount = 0
    
    def getActionProbs(self, *args, **kwargs):
        return np.random.dirichlet([0.5]*len(self.parent.children))

#####     BetaDog     #####
class BetaDog(PLAYER):
    def __init__(self, side=1, num_simulations:int=1000):
        super().__init__(side, name="BetaDog", mode="Search")
        # Board Size = 16*16
        self.INVALID = InvalidNode() # Preset Invalid Node
        self.StartNew()
        self.num_simulations = num_simulations
        #self.num_processes = 4 # Parallel Processes
        ### Hyper-parameters
        self.gamma = 0.95 # Discount Factor for Search
        self.c_puct = 1.5 # PUCT Exploration param: U(s,a) = c_puct * P(s,a) * sqrt(sum(N(s))) / (1+N(s,a))
        self.tem = 5.0 # Temperature param (1/tau): visitCounts = visitCounts ** self.tem
    
    def ACT(self, board:GomokuBoard, *args, **kwargs):
        """Take an action: Board -> (r,c)"""
        if not board.getSize() == 16:
            raise ValueError("Board Size must be 16*16")
        board = COMPRESS(board.getBoard()) # convert to CmprsBoard
        actionProbs = self.search(board)
        if True: action = np.random.choice(len(actionProbs), p=actionProbs)
        else: action = np.argmax(actionProbs)
        return action // 16, action % 16
    
    def StartNew(self):
        """Reset the root node and cache"""
        startBoard = CmprsBoard()
        self.Root = Node(startBoard) # Root Node
        self.Cache:dict[bytes,Node] = dict() # Cache for Searched Nodes: K-BoardKey, V-Node
        self.Cache[startBoard.toBytes()] = self.Root

    def search(self, board:CmprsBoard):
        """Search for the best action with MCTS"""
        key = board.toBytes()
        if not (key in self.Cache): self.Cache[key] = Node(board)
        start = self.Cache[key]

        for _ in range(self.num_simulations):
            self.simulate(start, board)

        return start.getActionProbs()
    
    def simulate(self, start:Node, board:CmprsBoard):
        """A simulation process"""
        node = start
        searchPath = [start]
        ### 选择动作：从起始节点向下选择，直至未扩展节点
        while node.haveChildren():
            _, node = self.selectChild(node)
            if not node.isValid: # 无效节点，中断搜索
                self.backup(0, searchPath)
                return
            searchPath.append(node)
            board = node.board
        ### 扩展回溯：对末端节点进行扩展，回溯更新节点价值
        if node.isWin: # 胜利节点，不扩展直接回溯更新
            self.backup(1, searchPath)
        else: # 普通节点，扩展并回溯更新
            self.expand(node, board)
            self.backup(0, searchPath)

    def selectChild(self, node:Node) -> tuple[int, Node]:
        """Select a child node (PUCT Alg)"""
        # score = Q(s,a) + U(s,a)
        # U(s,a) = c_puct * P(s,a) * sqrt(sum(N(s))) / (1+N(s,a))
        bestScore, bestAction, bestChild = -float('inf'), None, self.INVALID
        for pos, child in node.children.items():
            if not child.isValid: continue
            upper = self.c_puct * child.priorProb * np.sqrt(node.visitCount) / (1 + child.visitCount)
            score = child.getValue() + upper
            if score > bestScore: bestScore, bestAction, bestChild = score, pos, child
        return bestAction, bestChild

    def expand(self, node:Node, board:CmprsBoard):
        """expand the node (add child nodes)"""
        isBlack = board.nowBlack()
        P0 = 1 / (256 - board.getTotalMoves())
        for pos in range(256):
            if not board.isLegal(pos): # illegal move
                node.children[pos] = self.INVALID
            else: # legal move, check cache
                newBoard = board.placeStone(pos, isBlack)
                key = newBoard.toBytes()
                ### Check Cache
                if key in self.Cache: # already in cache
                    childNode = self.Cache[key]
                else: # not in cache, create a new node
                    if newBoard.checkWin(isBlack): childNode = Node(newBoard, 1, isWin=True)
                    else: childNode = Node(newBoard, P0, isWin=False)
                    self.Cache[key] = childNode
                node.children[pos] = childNode

    def backup(self, value, searchPath:list):
        """回溯更新节点价值。
        :param value: 当前节点的价值。
        :param searchPath: 搜索路径。
        """
        while searchPath:
            node:Node = searchPath.pop()
            node.visitCount += 1 # 更新访问次数
            node.valueSum += value # 更新累计价值
            value = -value * self.gamma # 计算对父节点的价值

    def getActionProbs(self, start:Node) -> np.ndarray:
        """获取动作概率分布pi (based on visitCounts)"""
        visitCounts = np.array([child.visitCount for child in start.children.values()])
        if self.tem != 1.0: visitCounts = visitCounts ** self.tem
        probs = visitCounts / max(np.sum(visitCounts), 1)
        return probs # [sz*sz]

if __name__ == "__main__":
    player1 = BetaDog(num_simulations=2000)
    player2 = BetaDog(num_simulations=2000)
    board = GomokuBoard(16)
    for i in range(10):
        a = player1.ACT(board)
        board.placeStone(a[0], a[1])
        b = player2.ACT(board)
        board.placeStone(b[0], b[1])
        print(a, b)
    print(board)