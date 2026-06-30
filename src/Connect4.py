import numpy as np
import pygame
import sys
import math
import time
import random

# ===== GAME SETTINGS =====
ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

SQUARESIZE = 100
RADIUS = SQUARESIZE // 2 - 5

# ===== Algorithm Selection =====
USE_ALPHA_BETA = False  # Set to False to use basic Minimax, True for Alpha-Beta Pruning
DEPTH = 3             # Change search depth (K) as you like

# ===== Nodes expanded global counter =====
nodes_expanded = 0

# ===== BOARD FUNCTIONS =====

def create_board():
    """Create a new empty board."""
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

def drop_piece(board, row, col, piece):
    """Place a piece at (row, col)."""
    board[row][col] = piece

def is_valid_location(board, col):
    """Check if a column has at least one empty cell."""
    return 0 <= col < COLUMN_COUNT and board[0][col] == 0

def get_next_open_row(board, col):
    """Get the lowest open row in a column. Returns -1 if column is full."""
    for r in range(ROW_COUNT - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1

def get_valid_locations(board):
    """Get all columns that still have space."""
    return [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]

def is_board_full(board):
    """Check if the board is full."""
    return len(get_valid_locations(board)) == 0

# ===== STRICT CONNECT 4 COUNTER =====
def strict_count_fours(board, piece):
    """
    Count the number of true connect-fours for given piece.
    Uses BFS to explore connected components.
    """
    visited = set()
    count = 0
    def bfs(sr, sc):
        queue = [(sr, sc)]
        component = [(sr, sc)]
        visited.add((sr, sc))
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROW_COUNT and 0 <= nc < COLUMN_COUNT:
                    if board[nr][nc] == piece and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                        component.append((nr, nc))
        return component
    def has_line_of_4(component):
        comp_set = set(component)
        for r, c in component:
            if all((r, c+i) in comp_set for i in range(4)):
                return True
            if all((r+i, c) in comp_set for i in range(4)):
                return True
            if all((r+i, c+i) in comp_set for i in range(4)):
                return True
            if all((r-i, c+i) in comp_set for i in range(4)):
                return True
        return False
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            if board[r][c] == piece and (r, c) not in visited:
                comp = bfs(r, c)
                if has_line_of_4(comp):
                    count += 1
    return count

# ===== WIN CHECKER =====
def winning_move(board, piece):
    """Check if the given piece has a winning (connect-4) line."""
    # Horizontal
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r][c+i] == piece for i in range(4)): return True
    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all(board[r+i][c] == piece for i in range(4)): return True
    # Positive diagonal \
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r+i][c+i] == piece for i in range(4)): return True
    # Negative diagonal /
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r-i][c+i] == piece for i in range(4)): return True
    return False

# ===== HEURISTIC =====
def evaluate_window(window, piece):
    """Give a heuristic score to a window of 4 cells."""
    score = 0
    opp = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 40
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 5
    if window.count(opp) == 3 and window.count(EMPTY) == 1:
        score -= 40
    return score

def score_position(board, piece):
    """Calculate the heuristic score for the entire board for a player."""
    score = 0
    center_col = list(board[:, COLUMN_COUNT//2])
    score += center_col.count(piece) * 6
    for r in range(ROW_COUNT):
        row = list(board[r])
        for c in range(COLUMN_COUNT - 3):
            window = row[c:c+4]
            score += evaluate_window(window, piece)
    for c in range(COLUMN_COUNT):
        col = list(board[:, c])
        for r in range(ROW_COUNT - 3):
            window = col[r:r+4]
            score += evaluate_window(window, piece)
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r+3-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    return score

# ===== MINIMAX with/without ALPHABETA & TREE-BUILD =====
def minimax_tree(board, depth, alpha, beta, maximizing, use_alpha_beta, build_tree=False):
    """
    Minimax with option for alpha-beta pruning.
    If build_tree=True, returns a tree structure which can be printed.
    Returns: (best_col, best_score, tree)
    """
    global nodes_expanded
    nodes_expanded += 1  # Increment nodes expanded at every node!

    valid_locations = get_valid_locations(board)
    is_terminal = winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(valid_locations) == 0

    if depth == 0 or is_terminal:
        score = None
        if winning_move(board, AI_PIECE):
            score = 1_000_000
        elif winning_move(board, PLAYER_PIECE):
            score = -1_000_000
        else:
            score = score_position(board, AI_PIECE)
        return None, score, {'col': None, 'score': score, 'children': []} if build_tree else None

    if maximizing:
        value = -math.inf
        best_col = random.choice(valid_locations) if valid_locations else None
        children = []
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row == -1:
                continue
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            _, new_score, child_tree = minimax_tree(
                b_copy, depth-1, alpha, beta, False, use_alpha_beta, build_tree)
            if build_tree:
                children.append(child_tree)
            if new_score > value:
                value = new_score
                best_col = col
            if use_alpha_beta:
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        return best_col, value, {'col': best_col, 'score': value, 'children': children} if build_tree else None
    else:
        value = math.inf
        best_col = random.choice(valid_locations) if valid_locations else None
        children = []
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row == -1:
                continue
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            _, new_score, child_tree = minimax_tree(
                b_copy, depth-1, alpha, beta, True, use_alpha_beta, build_tree)
            if build_tree:
                children.append(child_tree)
            if new_score < value:
                value = new_score
                best_col = col
            if use_alpha_beta:
                beta = min(beta, value)
                if alpha >= beta:
                    break
        return best_col, value, {'col': best_col, 'score': value, 'children': children} if build_tree else None

def print_minimax_tree(tree, depth=0):
    """Recursively print Minimax tree structure to console, with indentation."""
    if tree is None:
        return
    prefix = '  ' * depth
    print(f"{prefix}Col: {tree['col']} | Score: {tree['score']}")
    for child in tree.get('children', []):
        print_minimax_tree(child, depth+1)

# ===== GUI & ANIMATION =====
def draw_board(board, screen):
    """Draw the game board using pygame."""
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            pygame.draw.rect(screen,(0,0,255),
                             (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE,
                              SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen,(0,0,0),
                               (c*SQUARESIZE+SQUARESIZE//2,
                                r*SQUARESIZE+SQUARESIZE+SQUARESIZE//2),
                               RADIUS)
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen,(255,0,0),
                                   (c*SQUARESIZE+SQUARESIZE//2,
                                    r*SQUARESIZE+SQUARESIZE+SQUARESIZE//2),
                                   RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen,(255,255,0),
                                   (c*SQUARESIZE+SQUARESIZE//2,
                                    r*SQUARESIZE+SQUARESIZE+SQUARESIZE//2),
                                   RADIUS)
    pygame.display.update()

def animate_drop(screen, col, row, piece, board):
    """Visual animation for dropping a piece into the board."""
    color = (255,0,0) if piece == PLAYER_PIECE else (255,255,0)
    y = SQUARESIZE//2
    final_y = row*SQUARESIZE + SQUARESIZE + SQUARESIZE//2
    while y < final_y:
        pygame.draw.rect(screen,(0,0,0),(0,0,COLUMN_COUNT*SQUARESIZE,SQUARESIZE))
        draw_board(board, screen)
        pygame.draw.circle(screen,color,
                           (col*SQUARESIZE+SQUARESIZE//2, y),
                           RADIUS)
        pygame.display.update()
        y += 30
        time.sleep(0.01)

# ===== MAIN LOOP =====

board = create_board()
pygame.init()
pygame.font.init()
font = pygame.font.SysFont("monospace", 40)

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

screen = pygame.display.set_mode((width, height))
draw_board(board, screen)

turn = PLAYER
game_over = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        # === PLAYER MOVE ===
        if turn == PLAYER and not game_over:
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen,(0,0,0),(0,0,width,SQUARESIZE))
                posx = event.pos[0]
                pygame.draw.circle(screen,(255,0,0),(posx,SQUARESIZE//2),RADIUS)
                pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen,(0,0,0),(0,0,width,SQUARESIZE))
                col = event.pos[0] // SQUARESIZE
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    if row != -1:
                        animate_drop(screen, col, row, PLAYER_PIECE, board)
                        drop_piece(board, row, col, PLAYER_PIECE)
                        draw_board(board, screen)
                        # Important: do NOT set game_over here. Continue until board full.
                        turn = AI
        # === AI MOVE ===
        elif turn == AI and not game_over:
            valid = get_valid_locations(board)
            if len(valid) == 0:
                # nothing to do; will be handled below when board is full
                pass
            else:
                nodes_expanded = 0  # reset for this AI turn
                start = time.time()
                col, score, tree = minimax_tree(board, DEPTH, -math.inf, math.inf, True, USE_ALPHA_BETA, build_tree=True)
                end = time.time()
                print("\n===== Minimax Tree (AI turn) =====")
                print_minimax_tree(tree)
                print("Nodes expanded:", nodes_expanded)
                print("Time:", end - start, "seconds")

                # If minimax returned None (terminal or depth=0) or an invalid column,
                # pick randomly among valid columns so the board continues to fill.
                if col is None or not is_valid_location(board, col):
                    col = random.choice(valid)

                row = get_next_open_row(board, col)
                if row != -1:
                    animate_drop(screen, col, row, AI_PIECE, board)
                    drop_piece(board, row, col, AI_PIECE)
                    draw_board(board, screen)
                    # Important: do NOT set game_over here. Continue until board full.
                    turn = PLAYER

    # === WHEN BOARD FULL: Decide winner based on strict_count_fours ===
    if not game_over and is_board_full(board):
        p_score = strict_count_fours(board, PLAYER_PIECE)
        a_score = strict_count_fours(board, AI_PIECE)
        pygame.draw.rect(screen,(0,0,0),(0,0,width,SQUARESIZE))
        if p_score > a_score:
            msg = f"PLAYER WINS!  {p_score} vs {a_score}"
            color = (255,0,0)
        elif a_score > p_score:
            msg = f"AI WINS!  {a_score} vs {p_score}"
            color = (255,255,0)
        else:
            msg = f"DRAW!  {p_score} - {a_score}"
            color = (255,255,255)
        label = font.render(msg, True, color)
        screen.blit(label,(20,10))
        pygame.display.update()
        game_over = True

    if game_over:
        pygame.time.wait(6000)
        sys.exit()
