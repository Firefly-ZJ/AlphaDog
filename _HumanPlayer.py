#####     Player     #####
import pygame
from _Player import PLAYER

class HumanPl(PLAYER):
    """Human Player"""
    def __init__(self, side, name="机智如我"):
        super().__init__(side, name, mode="Human")
    
    def ACT(self, events:list, step, *args, **kwargs) -> tuple[int, int] | None:
        """Human Player's action
        Args:
            events: pygame events
            step: size of board grid
        Returns:
            action: (y,x) or None
        """
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_focused():
                x_pos, y_pos = pygame.mouse.get_pos()
                X, Y = (x_pos/step)-1 , (y_pos/step)-1
                X, Y = round(X), round(Y)
                return Y, X
        return None