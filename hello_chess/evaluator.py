import chess
import logging
import pickledb
from rich.progress import Progress
from random import shuffle

logger = logging.getLogger(__name__)

# fmt: off
PIECE_SQUARE_TABLES = {
    chess.PAWN: (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    chess.KNIGHT: ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    chess.BISHOP: ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    chess.ROOK: (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    chess.QUEEN: (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    chess.KING: (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# fmt: on

# class MoveEvaluator2001:
#     pass


class MoveEvaluator1998:
    def __init__(self, search_depth: int, progress: Progress):
        self.search_depth = search_depth
        self.progress = progress
        # self.db_cache_mm = pickledb.load("mm_cache.db", False)
        # self.db_cache_board_eval = pickledb.load("board_eval_cache.db", False)  moved to chessbot
        self.e4 = False
        self.board_evaluation = {"from_cache": 0, "evaluated": 0}
        self.cache_min_depth = 3
        self.total_trimmed_moves = 0
        self.caches_read = {x+1: 0 for x in range(3)}

        self.piece_value = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }

    def minimax(self, board, depth, alpha, beta, player, move_chain = None):
        if depth >= self.cache_min_depth:
            cache_key = board.fen() + " " + str(depth) + " " + str(player)
            cache_value = self.db_cache_mm.get(cache_key)
            if cache_value is not False:
                value = cache_value[0]
                best_move = chess.Move.from_uci(cache_value[1])
                # logger.info(f"Read from cache {cache_key}: {cache_value}")
                return value, best_move, []
        if depth == 0 or board.outcome():
            return self.evaluate_material(board), None, move_chain
        if player:
            color_modifier = 1
            value = -999999
        else:
            color_modifier = -1
            value = 999999
        if move_chain is None:
            move_chain = []
        # if depth == self.search_depth:
        #     print(f"Reasonable moves for white: {self.get_reasonable_moves(board)}")
        reasonable_moves = self.get_reasonable_moves(board)
        if depth == self.search_depth:
            logger.debug(f"Searching {len(list(reasonable_moves))} reasonable moves")
        if depth == self.search_depth:
            search_task = self.progress.add_task(
                f"Searching {len(reasonable_moves)} moves at depth {depth}",
                total=len(reasonable_moves),
            )
        for move in reasonable_moves:
            # if depth == self.search_depth:
            #     logger.debug(f"Trying move {move}")
            board.push(move)
            mm_val, _, nmc = self.minimax(board, depth - 1, alpha, beta, not player, move_chain + [move])
            if depth == self.search_depth:
                self.progress.update(search_task, advance=1)
            # determine whether this branch is higher value than the best one currently known
            # if mm_val * color_modifier > value:
            #     if depth == self.search_depth:
            #         print(f"found a better move for {'white' if player else 'black'}; {move} @ {mm_val}")
            #     value = mm_val
            #     best_move = move
            # alpha-beta pruning
            # logger.debug("mm_val is %s, value is %s, player is %s", mm_val, value, player)
            def print_found_better_move():
                if depth == self.search_depth:
                    logger.debug(f"L{depth}: Found a better move for {'white' if player else 'black'}; {move} @ {mm_val} - {nmc}")
                    
            if player:
                if mm_val > value:
                    print_found_better_move()
                    value = mm_val
                    best_move = move
                    best_move_chain = nmc
                if value >= beta:
                    board.pop()
                    self.total_trimmed_moves += 1
                    break
                if value > alpha:
                    # logger.debug("alpha %s -> %s", alpha, value)
                    alpha = value
            else:
                if mm_val < value:
                    print_found_better_move()
                    value = mm_val
                    best_move = move
                    best_move_chain = nmc
                if value <= alpha:
                    board.pop()
                    self.total_trimmed_moves += 1
                    break
                if value < beta:
                    # logger.debug("beta %s -> %s", beta, value)
                    beta = value

            # else:
            #     if depth == self.search_depth:
            #         print(f"eh white move; {move} @ {mm_val}")
            board.pop()
        if depth >= self.cache_min_depth:
            # logger.debug(f"Added to MM cache {cache_key}:{cache_value}")
            cache_value = [value, best_move.uci()]
            self.db_cache_mm.set(cache_key, cache_value)
        return value, best_move, best_move_chain

    def get_reasonable_moves(self, board):
        this_side = chess.WHITE if board.turn else chess.BLACK
        tie_moves = []
        continuing_moves = []  # these do not end the game
        # First we check for immediate game ending moves
        for move in board.legal_moves:
            # let's imagine we did make this move
            board.push(move)
            if board.outcome():
                # this move ends the game
                if board.outcome().winner == this_side:  # victory
                    board.pop()  # undo this move before we return
                    return [move]
                if not board.outcome().winner:
                    tie_moves.append(move)
                # elif board.outcome().winner != this_side:
                #     ...  # avoid losses
            else:
                continuing_moves.append(move)
            board.pop()  # undo move
        if not continuing_moves:
            return [tie_moves[0]]
        shuffle(continuing_moves)
        return continuing_moves

    # def get_move(self, board, current_depth) -> chess.Move:
    #     if (
    #         self.e4
    #         and board.fen()
    #         == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    #     ):
    #         return chess.Move.from_uci("e2e4")
    #     # move_eval = self.load_move_scores_cache(board)
    #     #     # use the cached results to quickly select the best move
    #     # if move_eval is not None:
    #     #     return max(move_eval, key=move_eval.get)
    #     this_side = chess.WHITE if board.turn else chess.BLACK
    #     tie_moves = []
    #     continuing_moves = []  # these do not end the game
    #     # First we check for immediate game ending moves
    #     for move in board.legal_moves:
    #         # let's imagine we did make this move
    #         board.push(move)
    #         if board.outcome():
    #             # this move ends the game
    #             if board.outcome().winner == this_side:  # victory
    #                 board.pop()  # undo this move before we return
    #                 return move
    #             if not board.outcome().winner:
    #                 tie_moves.append(move)
    #             # elif board.outcome().winner != this_side:
    #             #     ...  # avoid losses
    #         else:
    #             continuing_moves.append(move)
    #         board.pop()  # undo move
    #     if not continuing_moves:
    #         return tie_moves[0]
    #     # logger.debug(f"Evaluating {len(continuing_moves)} moves for {'White' if this_side else 'Black'}")
    #     move_eval = self.score_potential_moves(board, continuing_moves, current_depth)
    #     if current_depth == self.search_depth:
    #         ...
    #         # only cache results that have gone to maximum depth
    #         # self.save_move_scores_cache(board, move_eval)
    #         # move_eval = self.randomize_move_eval_score(move_eval)
    #     if this_side:
    #         move = max(move_eval, key=move_eval.get)
    #     else:
    #         move = min(move_eval, key=move_eval.get)
    #     return move

    # def load_move_scores_cache(self, board):
    #     move_eval = self.db_cache_move_scores.get(board.fen())
    #     if move_eval is not False:
    #         move_eval = {chess.Move.from_uci(k): v for k, v in move_eval.items()}
    #         return self.randomize_move_eval_score(move_eval)

    # def save_move_scores_cache(self, board, move_eval):
    #     serializable_move_eval = {k.uci(): v for k, v in move_eval.items()}
    #     self.db_cache_move_scores.set(board.fen(), serializable_move_eval)
    #     self.db_cache_move_scores.dump()

    # @staticmethod
    # def randomize_move_eval_score(move_eval):
    #     # add a small random amount to score so that when
    #     # we use max() on a bunch of options with the same
    #     # value it selects one randomly
    #     return {k: v + random() for k, v in move_eval.items()}

    # def score_potential_moves(
    #     self, board, potential_moves: list[chess.Move], remaining_depth: int
    # ) -> dict[chess.Move : float]:
    #     """Given a selection of moves, score them"""
    #     this_side = chess.WHITE if board.turn else chess.BLACK
    #     move_eval = {}  # map of moves to evaluated score
    #     if remaining_depth == self.search_depth:
    #         # create a nice progress bar while we look through moves
    #         search_task = self.progress.add_task(
    #             f"Searching {len(potential_moves)} moves at depth {remaining_depth}",
    #             total=len(potential_moves),
    #         )
    #     for move in potential_moves:
    #         # let's imagine we made this move
    #         board.push(move)
    #         # first evaluate material
    #         if remaining_depth == 0:
    #             score = self.evaluate_material(board)
    #         elif remaining_depth > 0:
    #             # what does our opponent's reponse look like?
    #             # we want our "score" for this position to reflect
    #             # the opponent's best move next turn
    #             o_move = self.get_move(board, remaining_depth - 1)
    #             # if that happened
    #             board.push(o_move)
    #             # what would the result be?
    #             score = self.evaluate_material(board)
    #             board.pop()
    #         # moves that cause check are a little better
    #         # if board.is_check():
    #         #     score += 5
    #         # attacking more squares is good; nudge score based on
    #         # total number of attacked squares to encourage development
    #         # diff_squares_attacked = self.get_len_attacked_squares(board) * (1 if this_side else -1)
    #         # #  = atks[this_side] - atks[not this_side]
    #         # score += diff_squares_attacked * 2
    #         # # threatenening enemy pieces of high value is good
    #         # atk_val = self.get_value_attacked_pieces(board)
    #         # diff_atk_val = atk_val[this_side] - atk_val[not this_side]
    #         # score += diff_atk_val * 5

    #         move_eval[move] = score
    #         board.pop()
    #         if remaining_depth == self.search_depth:
    #             self.progress.update(search_task, advance=1)
    #     return move_eval
    #     # logger.debug(move_eval)

    def evaluate_material(self, board: chess.Board) -> float:
        """Board evaluation is equal to material value minus opponent's"""
        if board.outcome():
            if board.outcome().winner is None:  # tie
                return 0
            if board.outcome().winner:
                return 20000
            else:
                return -20000
        fen = board.fen()
        ev = self.db_cache_board_eval.get(fen)
        if ev:
            self.board_evaluation["from_cache"] += 1
            return ev
        else:
            self.board_evaluation["evaluated"] += 1
        total_value = 0
        for color in (True, False):
            for piece_type, value in self.piece_value.items():
                pieces = board.pieces(piece_type, color)
                for pos in pieces:
                    index = (-pos - 1) if color else pos
                    total_value += PIECE_SQUARE_TABLES[piece_type][index] * (
                        1 if color else -1
                    )
                    # logger.debug("%s PST %s @ %s - %s", color, piece_type, pos,  PIECE_SQUARE_TABLES[piece_type][index])
                total_value += (len(pieces) * value) * (1 if color else -1)
        self.db_cache_board_eval.set(fen, total_value)
        return total_value

    # def get_len_attacked_squares(self, board: chess.Board) -> dict[bool: int]:
    #     """Given a board, return difference in number of attacked squares for each side
    #     Positive means White is attacking more squares"""
    #     cached_ta = self.load_attacked_sq_cache(board)
    #     if cached_ta:
    #         # logger.debug("Returned cached value %s", cached_ta)
    #         return cached_ta
    #     total_attacks_diff = 0
    #     for color in (True, False):
    #         for piece_type in range(2, 6):
    #             # exclude pawns and kings for optimization
    #             for square in board.pieces(piece_type, color):
    #                 total_attacks_diff += len(board.attacks(square)) * (1 if color else -1)
    #         # the following seems to take significantly longer
    #         # for attacker in chess.SquareSet(board.occupied_co[color]):
    #         #     attacks = board.attacks(attacker)
    #         #     total_attacks[color] += len(attacks)
    #     self.save_attacked_sq_cache(board, total_attacks_diff)
    #     return total_attacks_diff

    # def get_value_attacked_pieces(self, board: chess.Board) -> dict[bool: int]:
    #     """Value of pieces which are under threat by the other side"""
    #     attacked_value = {
    #         True: 0,
    #         False: 0
    #     }
    #     for color in (True, False):
    #         for piece_type in range(2, 6):
    #             for square in board.pieces(piece_type, color):
    #                 if board.is_attacked_by(not color, square):
    #                     attacked_value[color] += self.piece_value[piece_type]
    #     return attacked_value
