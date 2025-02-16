from dataclasses import dataclass

from enum import Enum

import math

class Alliance(Enum):
    RED = 0, "Red"
    BLUE = 1, "Blue"

class Direction(Enum):
    LEFT = 0
    RIGHT = 1

class Reef:

    class Branch(Enum):
        A = 0, "A"
        B = 1, "B"
        C = 2, "C"
        D = 3, "D"
        E = 4, "E"
        F = 5, "F"
        G = 6, "G"
        H = 7, "H"
        I = 8, "I"
        J = 9, "J"
        K = 10, "K"
        L = 11, "L"

    class CoralState(Enum):
        OFF = 0
        ON = 1
    
    class Level(Enum):
        L2 = 0
        L3 = 1
        L4 = 2

    global BLUE_ALLIANCE_TAGS
    global RED_ALLIANCE_TAGS 
    global BRANCHES

    BLUE_ALLIANCE_TAGS = [18, 17, 22, 21, 20, 19]
    RED_ALLIANCE_TAGS = [7, 8, 9, 10, 11, 6]
    BRANCHES = [branch for branch in Branch]
    # TAG INDEXES [A, B, C, D, E, F, G, H, I, J, K, L]
    
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
        for branch in BRANCHES:
            self.branch_to_tag.update({branch : self.alliance_tags[index]})
            char_index += 1
            if char_index % 2 == 0:
                index += 1
    
    def init_branch_states(self):
        #Initialize Branch States:
        self.branch_state = {}
        for branch in BRANCHES:
            # Initialize L2, L3, L4
            self.branch_state.update({branch : {
                self.Level.L2: self.CoralState.OFF,
                self.Level.L3 : self.CoralState.OFF,
                self.Level.L4 : self.CoralState.OFF}})
    
    def get_all_states(self):
        return self.branch_state

    # Get the state of the branches
    def get_branch(self, branch : Branch):
        return self.branch_state.get(branch)

    # get_branch_state_at("A", Reef.Level.L2) => Reef.BranchState.OFF
    def get_branch_state_at(self, branch : Branch, level : Level):
        branch_face = self.get_branch(branch)
        print("branch", branch_face)
        return branch_face.get(level)
    
    # get_branches_at_tag(7) => ["A", "B"]
    def get_branches_at_tag(self, id : int):
        if id in self.alliance_tags:
           index = self.alliance_tags.index(id) * 2
           return BRANCHES[index:index+2]
        return -1
    
    # get_tag_from_branch("A") => 7
    def get_tag_from_branch(self, branch : chr):
        index = int(math.floor(branch.value[0] / 2)) # Retrieves the index
        return self.alliance_tags[index]
    
    # toggle_branch(Reef.Branch.A, Reef.Level.L1) => sets to true
    def set_branch_state(self, branch : Branch, level : Level, state : CoralState):
        self.branch_state[branch][level] = state

    # get_branch_with_state(CoralState.ON) => [A, B, C] which contains CoralState.ON
    def get_branch_with_state(self, state : CoralState):
        pass

    def printBranchList(self):
        print(BRANCHES)
        

red = Reef(Alliance.RED)
blue = Reef(Alliance.BLUE)
#print("=======RED======")
#for x in range(6, 12):
#    print(x, red.get_branches_at_tag(x))

#print("=======BLUE======")
#for x in range(17, 22):
#    print(x, blue.get_branches_at_tag(x))
#print(test.get_branch_state_at('A', Reef.Level.L2))
#print(red.get_all_states())

#for x in Reef.Branch:
#    print(x, red.get_tag_from_branch(x))

    # robot.goToBranch(L1, l2)
    # goToReefBranch(A = Branch, L1 = Level)

print(red.get_branch_state_at(Reef.Branch.A, Reef.Level.L2))
red.set_branch_state(Reef.Branch.A, Reef.Level.L2, Reef.CoralState.OFF)
print(red.get_branch_state_at(Reef.Branch.A, Reef.Level.L2))