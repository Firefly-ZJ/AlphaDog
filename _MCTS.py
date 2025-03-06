#####     Monte-Carlo Tree Search     #####
import numpy as np
import torch

#####     Node     #####
class Node():
    def __init__(self, priorProb:float=1.0, isWin:bool=False):
        """Node of M-C Search Tree"""
        self.isValid = True # 有效节点
        self.children = dict()  # 子节点: key-Position, value-ChildNode
        self.visitCount = 0  # 该节点的访问次数
        self.valueSum = 0  # 该节点的累计价值
        self.priorProb = priorProb  # 该节点的先验概率
        self.isWin = isWin  # 该节点是否为胜利节点

    def haveChildren(self) -> bool:
        """Whether is already expanded (has children)"""
        return len(self.children) > 0

    def getValue(self) -> float:
        """Mean value of the node: valueSum/visitCount"""
        if self.visitCount == 0: return 0
        else: return self.valueSum / self.visitCount
    
    def getActionProbs(self) -> np.ndarray:
        """action probs of children: based on visitCounts"""
        visitCounts = np.array([child.visitCount for child in self.children.values()])
        probs = visitCounts / np.sum(visitCounts)
        return probs # [sz*sz]

class InvalidNode(Node):
    __slots__ = ("visitCount", "isValid")
    def __init__(self):
        """Invalid Move Node (no children)"""
        self.isValid = False # 无效节点
        self.visitCount = 0
    
    def getActionProbs(self):
        raise RuntimeError("Select action from Invalid Node")

#####     MCTS     #####
class MCTS():
    def __init__(self, model, num_simulations:int=200, device=None):
        """Monte-Carlo Tree Search
        Args:
            model: Policy-Value Network
            num_simulations: simulations per search
        """
        self.Model = model # Policy-Value Network
        self.Root = Node() # Root Node
        self.INVALID = InvalidNode() # Preset Invalid Node
        self.dv = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Simulations = num_simulations # Simulations per Search
        ### Hyper-parameters
        self.gamma = 0.99 # Discount Factor for Future Rewards
        self.c_puct = 5 # PUCT Exploration param: U(s,a) = c_puct * P(s,a) * sqrt(sum(N(s))) / (1+N(s,a))
    
    def setHyper(self, **kwargs):
        """:param kwargs: Hypers (simu, c_puct, noise, eps, tem)"""
        if "simu" in kwargs: self.Simulations = kwargs["simu"]
        if "c_puct" in kwargs: self.c_puct = kwargs["c_puct"]
        if "gamma" in kwargs: self.gamma = kwargs["gamma"]
    
    def reset(self):
        """Reset Root, and start a new game"""
        self.Root = Node()
    
    def evalState(self, state:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """State -> Policy, Value"""
        return self.Model.forward(state)
    
    def search(self, board) -> np.ndarray:
        """Search for action probs by MCTS
        Args:
            board: 当前棋盘
        Returns:
            动作概率分布（根据子节点的访问次数计算）
        """
        start:Node = self.Root # 起始节点
        for move in board.moves:
            pos = move[0] * board.getSize() + move[1]
            if not start.haveChildren(): self.expand(start, pos, board)
            start = start.children[pos]
            if not start.isValid: raise ValueError("Invalid Move Recorded")
        
        for _ in range(self.Simulations):
            self.simulate(start, board)
        
        return start.getActionProbs()
    
    def simulate(self, start:Node, board):
        """A simulation process"""
        node = start
        board_ = board.copy()
        searchPath = [start]
        # Select till leaf
        while node.haveChildren():
            pos, node = self.selectChild(node)
            if not node.isValid: # No Valid Child
                self.backup(0, searchPath)
                return
            searchPath.append(node)
            board_.placeStone(pos//board_.Size, pos%board_.Size)
        # Expand and Backup
        if node.isWin:
            self.backup(1, searchPath)
        else:
            policy, value = self.evalState(board_.getStateAsT().to(self.dv))
            policy = torch.exp(policy.squeeze()) # logSoftmax -> Prob
            self.expand(node, policy, board_)
            self.backup(-value.item(), searchPath) # -Value

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

    def expand(self, node:Node, policy:torch.Tensor|int, board):
        """Expand the node (add child nodes)"""
        size = board.getSize()
        if type(policy) != torch.Tensor: # Assigned Policy
            ### 注：实际为np.int64
            for pos in range(size*size):
                if pos == policy: node.children[pos] = Node(priorProb=1.0)
                else: node.children[pos] = self.INVALID
        else: # Estimated Policy
            for pos in range(size*size):
                if board.isLegal(pos//size, pos%size): # legal move
                    isWin = board.isWin(pos//size, pos%size)
                    node.children[pos] = Node(priorProb=policy[pos].item(), isWin=isWin)
                else:
                    node.children[pos] = self.INVALID

    def backup(self, value, searchPath):
        """Update the value of nodes backward"""
        while searchPath:
            node:Node = searchPath.pop()
            node.visitCount += 1  # 更新访问次数
            node.valueSum += value  # 更新累计价值
            value = -value * self.gamma  # 计算对父节点的价值

if __name__ == "__main__":
    from _AlphaDog import *
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node = Node()
    mcts = MCTS(GomokuNet().to(device), 400, device=device)
    board = GomokuBoard()
    print(mcts.search(board))
    board.placeStone(0, 0)
    print(mcts.search(board))
    board.placeStone(0, 2)
    print(mcts.search(board))
