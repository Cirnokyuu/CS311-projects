import numpy as np
import random
import time
from math import sqrt
from functools import lru_cache

DEBUG = True
OUTPUT = False

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
SIZE = 8
CORNER_VALUE = -13
STAR_VALUE = -1.2
C_VALUE = 4
MID_C_VALUE = -1.5
COR_E_VALUE = -0.9

random.seed(0)

DIRECTIONS = [
    (1, 0xFEFEFEFEFEFEFEFE),
    (-1, 0x7F7F7F7F7F7F7F7F),
    (8, 0xFFFFFFFFFFFFFF00),
    (-8, 0x00FFFFFFFFFFFFFF),
    (9, 0xFEFEFEFEFEFEFE00),
    (7, 0x7F7F7F7F7F7F7F00),
    (-7, 0x00FEFEFEFEFEFEFE),
    (-9, 0x007F7F7F7F7F7F7F)
]

def to_mask(x: int, y: int) -> int:
    return 1 << (x * 8 + y)

CORNERS = [to_mask(0,0), to_mask(0,7), to_mask(7,7), to_mask(7,0)]
STAR_POSITIONS = [to_mask(1,1), to_mask(1,6), to_mask(6,6), to_mask(6,1)]

C_POSITIONS = [
    [to_mask(1,0), to_mask(0,1)],      # (1,0), (0,1)
    [to_mask(1,7), to_mask(0,6)],      # (1,7), (0,6)
    [to_mask(6,7), to_mask(7,6)],      # (6,7), (7,6)
    [to_mask(6,0), to_mask(7,1)]       # (6,0), (7,1)
]

MID_C_MASK = (  to_mask(1,2) | to_mask(2,1) |
                to_mask(1,5) | to_mask(2,6) |
                to_mask(5,1) | to_mask(6,2) |
                to_mask(5,6) | to_mask(6,5) )

COR_E_POSITIONS = [
    [to_mask(2,0), to_mask(0,2)],
    [to_mask(2,7), to_mask(0,5)],
    [to_mask(5,7), to_mask(7,5)],
    [to_mask(5,0), to_mask(7,2)]
]

hash_results = {}

def get_legal_moves(my, opponent):
    empty = ~(my | opponent)
    legal = 0
    for shift, mask in DIRECTIONS:
        if shift > 0:
            initial = (my << shift) & mask
        else:
            initial = (my >> -shift) & mask
        candidates = opponent & initial
        while candidates:
            if shift > 0:
                next_step = (candidates << shift) & mask
            else:
                next_step = (candidates >> -shift) & mask
            legal |= next_step & empty
            candidates = next_step & opponent
    return legal

def compute_flips(pos, my, opponent):
    flips = 0
    for shift, mask in DIRECTIONS:
        current = 1 << pos
        temp = 0
        while True:
            if shift > 0:
                current = (current << shift) & mask
            else:
                current = (current >> -shift) & mask
            if not (current & opponent):
                break
            temp |= current
        if current & my:
            flips |= temp
    return flips

def board_to_bits(board):
    black = 0
    white = 0
    for i in range(SIZE):
        for j in range(SIZE):
            pos = i * SIZE + j
            if int(board[i,j]) == COLOR_BLACK:
                black |= 1 << pos
            elif int(board[i,j]) == COLOR_WHITE:
                white |= 1 << pos
    return black, white

def bits_to_board(black, white):
    board = np.zeros((SIZE, SIZE), dtype=int)
    for i in range(SIZE):
        for j in range(SIZE):
            pos = i * SIZE + j
            if black & (1 << pos):
                board[i,j] = COLOR_BLACK
            elif white & (1 << pos):
                board[i,j] = COLOR_WHITE
    return board

BIT_COUNT_TABLE = [bin(x).count('1') for x in range(65536)]

def count_bits(x):
    return (BIT_COUNT_TABLE[x & 0xFFFF] + BIT_COUNT_TABLE[(x >> 16) & 0xFFFF] +
            BIT_COUNT_TABLE[(x >> 32) & 0xFFFF] + BIT_COUNT_TABLE[(x >> 48) & 0xFFFF])

def calc_special_score(black, white, color, tim_score, my_mov, op_mov):
    sp_score = 0
    my_corner = CORNER_VALUE * (0.55 + my_mov / 10)
    op_corner = CORNER_VALUE * (0.55 + op_mov / 10)
    
    corner_is_full = [
        (black | white) & CORNERS[0],
        (black | white) & CORNERS[1],
        (black | white) & CORNERS[2],
        (black | white) & CORNERS[3]
    ]
    
    for i in range(4):
        if corner_is_full[i]:
            corner_color = 1 if (white & CORNERS[i]) else -1
            if corner_color == color:
                sp_score += my_corner * corner_color
            else:
                sp_score += op_corner * corner_color
        else:
            cnt_cp = 0
            # for cp in C_POSITIONS[i]:
            #     if (black | white) & cp:
            #         cnt_cp += 1
            #         cp_color = 1 if (white & cp) else -1
            #         sp_score += C_VALUE * cp_color
            for j in range(2):
                cp = C_POSITIONS[i][j]
                if (black | white) & cp:
                    cnt_cp += 1
                    cp_color = 1 if (white & cp) else -1
                    sp_score += C_VALUE * cp_color
                elif (black | white) & COR_E_POSITIONS[i][j]:
                    cor_color = 1 if (white & COR_E_POSITIONS[i][j]) else -1
                    sp_score += COR_E_VALUE * cor_color
            star_val = STAR_VALUE * (1 - cnt_cp)
            if (black | white) & STAR_POSITIONS[i]:
                star_color = 1 if (white & STAR_POSITIONS[i]) else -1
                sp_score += star_val * star_color

    mid_white = white & MID_C_MASK
    mid_black = black & MID_C_MASK
    e_score = (count_bits(mid_white) - count_bits(mid_black)) * MID_C_VALUE * (1 - 0.4 * tim_score)

    return (sp_score + e_score) * (1 - 0.35 * tim_score * tim_score)

@lru_cache(maxsize=2**18)
def evaluate(black, white, color):
    cnt_tim = count_bits(black | white)
    sum_score = count_bits(white) - count_bits(black)
    tim_score = cnt_tim / 64.0
    
    my = white if color == 1 else black
    opponent = black if color == 1 else white
    
    my_moves = get_legal_moves(my, opponent)
    my_move = count_bits(my_moves)
    op_move = count_bits(get_legal_moves(opponent, my))
    move_score = my_move - op_move
    
    if color == -1:
        move_score = -move_score
    
    if my_move == 0:
        if op_move == 0:
            return -sum_score * 1000
        else:
            return evaluate(black, white, -color)
    
    sp_score = calc_special_score(black, white, color, tim_score, my_move, op_move)
    score = (1.2 - 0.55 * tim_score**2 * sqrt(tim_score)) * move_score - (0.2 + 0.8 * tim_score) * sum_score + sp_score
    
    if my_move == 0:
        score += 1 + 7 * tim_score * color
    
    return score

@lru_cache(maxsize=2**18)
def get_actions(black: int, white: int, color: int):
    my = white if color == 1 else black
    opponent = black if color == 1 else white
    
    legal_moves = get_legal_moves(my, opponent)
    
    actions = []
    for pos in range(64):
        if legal_moves & (1 << pos):
            actions.append((pos//8, pos%8))
    
    return tuple(actions)

class othello():
    def __init__(self, board, color):
        self.black, self.white = board
        self.color = color
        self.hash_value = self.__hash__()
    
    def actions(self):
        return list(get_actions(self.black, self.white, self.color))
    
    def result(self, x, y):
        if x == -1 and y == -1:
            return othello((self.black, self.white), -self.color)
        
        pos = x * 8 + y
        my = self.white if self.color == 1 else self.black
        opponent = self.black if self.color == 1 else self.white
        
        flips = compute_flips(pos, my, opponent)
        new_my = my | (1 << pos) | flips
        new_opponent = opponent ^ flips
        
        if self.color == 1:
            new_black = new_opponent
            new_white = new_my
        else:
            new_black = new_my
            new_white = new_opponent
        
        return othello((new_black, new_white), -self.color)
    
    def game_over(self):
        if self.get_round() == 64:
            return True
        if len(self.actions()) == 0 and len(self.result(-1, -1).actions()) == 0:
            return True
        return False

    def get_results(self):
        if self.hash_value in hash_results:
            return hash_results[self.hash_value]
        
        results = []
        if len(self.actions()) == 0:
            res_opponent = self.result(-1, -1)
            if(len(res_opponent.actions()) == 0):
                return results
            else:
                results.append(res_opponent)
        else:
            for act in self.actions():
                results.append(self.result(act[0], act[1]))
        
        hash_results[self.hash_value] = results
        return results
    
    def __hash__(self):
        return hash((self.black & 0xFFFFFFFFFFFFFFFF, self.white & 0xFFFFFFFFFFFFFFFF, self.color))
    
    def __eq__(self, other):
        return (self.black == other.black and self.white == other.white and self.color == other.color)
    
    def get_round(self):
        return count_bits(self.black | self.white)
    
    def evaluate(self):
        # if OUTPUT:
        #     # print(self.board)
        #     for i in range(SIZE):
        #         for j in range(SIZE):
        #             pos = i * SIZE + j
        #             if self.black & (1 << pos):
        #                 print("B", end=" ")
        #             elif self.white & (1 << pos):
        #                 print("W", end=" ")
        #             else:
        #                 print(".", end=" ")
        #         print()

        #     print("now_color = ", self.color)
        #     print("my_move = ", my_move, "op_move = ", op_move)
        #     print("move_score = ", move_score, "sp_score = ", sp_score, "sum = ", sum_score ,"score = ", score)

        return evaluate(self.black, self.white, self.color)

class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.start_time = time.time()
        self.timer = 0
        self.time_over = 0
        self.candidate_list = []
        self.top_n = 5
    
    def check_timeout(self):
        self.timer = self.timer + 1
        if self.timer >= 16:
            self.timer = 0
            self.time_over = time.time() - self.start_time >= self.time_out * 0.93
        return self.time_over

    def max_value(self, board, alpha, beta, depth):
        if self.check_timeout():
            return -float('inf')
        if depth == 0 or board.game_over():
            return board.evaluate()
        v = -float('inf')
        nxt_list = sorted(board.get_results(), key=lambda bd: bd.evaluate(), reverse=True)[:self.top_n]
        if len(nxt_list) == 0:
            return board.evaluate()
        for act in nxt_list:
            v = max(v, self.min_value(act, alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, board, alpha, beta, depth):
        if self.check_timeout():
            return float('inf')
        if depth == 0 or board.game_over():
            return board.evaluate()
        v = float('inf')
        nxt_list = sorted(board.get_results(), key=lambda bd: bd.evaluate())[:self.top_n]
        if len(nxt_list) == 0:
            return board.evaluate()
        for act in nxt_list:
            v = min(v, self.max_value(act, alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def go(self, chessboard):
        self.start_time = time.time()
        self.time_over = 0
        self.candidate_list.clear()
        board = othello(board_to_bits(chessboard), self.color)

        # if board.hash_value in OPENING_LIBRARY:
        #     act = OPENING_LIBRARY[board.hash_value]
        #     self.candidate_list = [act]
        # else:
        
        self.candidate_list = board.actions()
        act_list = self.candidate_list.copy()
        

        # print("num of my actions = ", len(self.candidate_list), "num of opponent actions = ", len(board.result(-1, -1).actions()))

        if len(self.candidate_list) == 0:
            return
        true_act = self.candidate_list[0]
        now_round = board.get_round()
        dol = 0
        upl = (66 - now_round) * 2
        basic_topn = 3 + (1 if now_round<=28 else 0) + (1 if now_round<=18 else 0)
        # upl = 1
        if self.color == 1:
            act_list = sorted(act_list, key=lambda act: board.result(act[0], act[1]).evaluate(), reverse=True)
            for depth in range(dol, upl):
                self.top_n = basic_topn + max(0, depth + now_round - 64)
                best = -float('inf')
                best_act = self.candidate_list[0]
                for act in act_list:
                    # print("act = ", act)
                    v = self.min_value(board.result(act[0], act[1]), best, float('inf'), depth)
                    if v > best:
                        best = v
                        best_act = act
                if self.check_timeout():
                    break
                true_act = best_act
                self.candidate_list.append(true_act)
                if DEBUG and depth >= 7:
                    print(time.time()-self.start_time, "  depth = ", depth, "best_act = ", best_act, "best = ", best)
        else:
            act_list = sorted(act_list, key=lambda act: board.result(act[0], act[1]).evaluate())
            for depth in range(dol, upl):
                self.top_n = basic_topn + max(0, depth + now_round - 64)
                best = float('inf')
                best_act = self.candidate_list[0]
                for act in act_list:
                    # print("act = ", act)
                    v = self.max_value(board.result(act[0], act[1]), -float('inf'), best, depth)
                    if v < best:
                        best = v
                        best_act = act
                if self.check_timeout():
                    break
                true_act = best_act
                self.candidate_list.append(true_act)
                if DEBUG and depth >= 7:
                    print(time.time()-self.start_time, "  depth = ", depth, "best_act = ", best_act, "best = ", best)
        
    