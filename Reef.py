from dataclasses import dataclass

from enum import Enum

class Alliance(Enum):
    RED = 0, "Red"
    BLUE = 1, "Blue"


class Reef:

    class State(Enum):
        OFF = 0,
        ON = 1
    
    class Level(Enum):
        L2 = 0, # Index 
        L3 = 1,
        L4 = 2

    # A, B, C, D, E, F, G, H, I, J, K, L
    global BLUE_ALLIANCE_TAGS
    global RED_ALLIANCE_TAGS 
    BLUE_ALLIANCE_TAGS = [18, 17, 22, 21, 20, 19]
    RED_ALLIANCE_TAGS = [7, 8, 9, 10, 11, 6]
    def __init__(self, alliance : Alliance):
        self.alliance = alliance
        assert self.alliance == Alliance.RED or self.alliance == Alliance.BLUE, "Alliance Not Properly Initialized"

        self.branch_state = {
            "A" : self.State.OFF,
            "B" : self.State.OFF,
            "C" : self.State.OFF,
            "D" : self.State.OFF,
            "E" : self.State.OFF,
            "F" : self.State.OFF,
            "G" : self.State.OFF,
            "H" : self.State.OFF,
            "I" : self.State.OFF,
            "J" : self.State.OFF,
            "K" : self.State.OFF,
            "L" : self.State.OFF,
        }

        if self.alliance == Alliance.RED:
            branch_to_tag = {
                "A": RED_ALLIANCE_TAGS[0], "B": RED_ALLIANCE_TAGS[0],
                "C": RED_ALLIANCE_TAGS[1], "D": RED_ALLIANCE_TAGS[1],
                "E": RED_ALLIANCE_TAGS[2], "F": RED_ALLIANCE_TAGS[2],
                "G": RED_ALLIANCE_TAGS[3], "H": RED_ALLIANCE_TAGS[3],
                "I": RED_ALLIANCE_TAGS[4], "J": RED_ALLIANCE_TAGS[4],
                "K": RED_ALLIANCE_TAGS[5], "L": RED_ALLIANCE_TAGS[5],
            }
        else: # Blue Alliance
            branch_to_tag = {
                "A": BLUE_ALLIANCE_TAGS[0], "B": BLUE_ALLIANCE_TAGS[0],
                "C": BLUE_ALLIANCE_TAGS[1], "D": BLUE_ALLIANCE_TAGS[1],
                "E": BLUE_ALLIANCE_TAGS[2], "F": BLUE_ALLIANCE_TAGS[2],
                "G": BLUE_ALLIANCE_TAGS[3], "H": BLUE_ALLIANCE_TAGS[3],
                "I": BLUE_ALLIANCE_TAGS[4], "J": BLUE_ALLIANCE_TAGS[4],
                "K": BLUE_ALLIANCE_TAGS[5], "L": BLUE_ALLIANCE_TAGS[5],
            }

        branch_state = {
            "A" : {self.Level.L2: self.State.OFF}
        }

        

    def toggle_branch(self, )