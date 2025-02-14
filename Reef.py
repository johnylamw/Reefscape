from dataclasses import dataclass

from enum import Enum

class Alliance(Enum):
    RED = 0, "Red"
    BLUE = 1, "Blue"

class Direction(Enum):
    LEFT = 0
    RIGHT = 1

class Reef:

    class BranchState(Enum):
        OFF = 0
        ON = 1
    
    class Level(Enum):
        L2 = 0
        L3 = 1
        L4 = 2

    global BLUE_ALLIANCE_TAGS
    global RED_ALLIANCE_TAGS 
    global BRANCHES_COL
    # TAG INDEXES [A, B, C, D, E, F, G, H, I, J, K, L]
    BLUE_ALLIANCE_TAGS = [18, 17, 22, 21, 20, 19]
    RED_ALLIANCE_TAGS = [7, 8, 9, 10, 11, 6]
    BRANCHES_COL = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    
    def __init__(self, alliance : Alliance):
        self.alliance = alliance
        assert self.alliance == Alliance.RED or self.alliance == Alliance.BLUE, "Alliance Not Properly Initialized"

        # Initialize
        self.init_alliance_settings()
        self.init_branch_to_tag()
        self.init_branch_states()
        
    def init_alliance_settings(self):
        # Initialize Alliance Specific Settings
        if self.alliance == Alliance.RED:
            self.alliance_tags = RED_ALLIANCE_TAGS
        elif self.alliance == Alliance.BLUE:
            self.alliance_tags = BLUE_ALLIANCE_TAGS

    def init_branch_to_tag(self):
        # Branch Char -> Tag Dictionary 
        self.branch_to_tag = {}
        index = 0
        char_index = 0
        for col in BRANCHES_COL:
            self.branch_to_tag.update({col : self.alliance_tags[index]})
            char_index += 1
            if char_index % 2 == 0:
                index += 1
    
    def init_branch_states(self):
        #Initialize Branch States:
        self.branch_state = {}
        for col in BRANCHES_COL:
            # Initialize L2, L3, L4
            self.branch_state.update({col : {
                self.Level.L2: self.BranchState.OFF,
                self.Level.L3 : self.BranchState.OFF,
                self.Level.L4 : self.BranchState.OFF}})
    
    def get_all_states(self):
        return self.branch_state

    # Get the state of the branches
    def get_branch_column(self, col : chr):
        return self.branch_state.get(col)

    def get_branch_state_at(self, col : chr, level : Level):
        column = self.get_branch_column(col)
        return column.get(level)
    
    def get_branches_at_tag(self, id : int):
        pass
    
    def get_tag_from_col(self, col : chr):
        pass
    #def toggle_branch(self, )

#test = Reef(Alliance.RED)
#print(test.get_branch_state_at('A', Reef.Level.L2))
#print(test.get_all_states())
