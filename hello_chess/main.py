import chess
from random import choice, random
import logging
from rich.logging import RichHandler
from rich.progress import Progress
import chess.svg
import pickledb
import ast

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


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


class MoveEvaluator1998:
    def __init__(self, search_depth: int, progress: Progress):
        self.search_depth = search_depth
        self.progress = progress
        self.db_cache_board_eval = pickledb.load('board_eval_cache.db', False)
        self.db_cache_attacked_sq = pickledb.load('attacked_square_cache.db', False)
        self.e4 = False

        self.piece_value = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }
    
    def get_move(self, board, current_depth) -> chess.Move:
        if board.fen() == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' and self.e4:
            return chess.Move.from_uci("e2e4")
        move_eval = self.load_eval_cache(board)
            # use the cached results to quickly select the best move
        if move_eval is not None:
            return max(move_eval, key=move_eval.get)
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
                    return move
                if not board.outcome().winner:
                    tie_moves.append(move)
                # elif board.outcome().winner != this_side:
                #     ...  # avoid losses
            else:
                continuing_moves.append(move)
            board.pop()  # undo move
        if not continuing_moves:
            return tie_moves[0]
        # logger.debug(f"Evaluating {len(continuing_moves)} moves for {'White' if this_side else 'Black'}")
        move_eval = self.score_potential_moves(board, continuing_moves, current_depth)
        if current_depth == self.search_depth:
            # only cache results that have gone to maximum depth
            self.save_eval_cache(board, move_eval)
            move_eval = self.randomize_eval_score(move_eval)
        move = max(move_eval, key=move_eval.get)  # return the best move
        return move
    
    def load_eval_cache(self, board):
        move_eval = self.db_cache_board_eval.get(board.fen())
        if move_eval is not False:
            move_eval = {chess.Move.from_uci(k): v for k, v in move_eval.items()}
            return self.randomize_eval_score(move_eval)

    def save_eval_cache(self, board, move_eval):
        serializable_move_eval = {k.uci(): v for k, v in move_eval.items()}
        self.db_cache_board_eval.set(board.fen(), serializable_move_eval)
        self.db_cache_board_eval.dump()
    
    @staticmethod
    def randomize_eval_score(move_eval):
        # add a small random amount to score so that when
        # we use max() on a bunch of options with the same
        # value it selects one randomly
        return {k: v + random() for k, v in move_eval.items()}

    def score_potential_moves(self, board, potential_moves: list[chess.Move], remaining_depth: int) -> dict[chess.Move: float]:
        """Given a selection of moves, score them"""
        this_side = chess.WHITE if board.turn else chess.BLACK
        move_eval = {}  # map of moves to evaluated score
        if remaining_depth == self.search_depth:
            # create a nice progress bar while we look through moves
            search_task = self.progress.add_task(f"Searching {len(potential_moves)} moves at depth {remaining_depth}", total=len(potential_moves))
        for move in potential_moves:
            # let's imagine we made this move
            board.push(move)
            # first evaluate material
            if remaining_depth == 0:
                score = self.evaluate_material(board, this_side)
            elif remaining_depth > 0:
                # what does our opponent's reponse look like?
                # we want our "score" for this position to reflect
                # the opponent's best move next turn
                o_move = self.get_move(board, remaining_depth - 1)
                # if that happened
                board.push(o_move)
                # what would the result be?
                score = self.evaluate_material(board, this_side)
                board.pop()
            # moves that cause check are a little better
            if board.is_check():
                score += 5
            # attacking more squares is good; nudge score based on
            # total number of attacked squares to encourage development
            # diff_squares_attacked = self.get_len_attacked_squares(board) * (1 if this_side else -1)
            # #  = atks[this_side] - atks[not this_side]
            # score += diff_squares_attacked * 2
            # # threatenening enemy pieces of high value is good
            # atk_val = self.get_value_attacked_pieces(board)
            # diff_atk_val = atk_val[this_side] - atk_val[not this_side]
            # score += diff_atk_val * 5

            move_eval[move] = score
            board.pop()
            if remaining_depth == self.search_depth:
                self.progress.update(search_task, advance=1)
        return move_eval
        # logger.debug(move_eval)

    def load__cache(self, board):
        return self.db_cache_attacked_sq.get(board.fen())

    def save_attacked_sq_cache(self, board, total_attacks_diff):
        self.db_cache_attacked_sq.set(board.fen(), total_attacks_diff)


        
    def evaluate_material(self, board: chess.Board, side: bool) -> float:
        """Board evaluation is equal to material value minus opponent's"""
        if board.outcome():
            if not board.outcome().winner:  # tie
                return 0
            if board.outcome().winner == side:
                return 2000
            else:
                return -2000
        total_value = {
            chess.WHITE: 0,
            chess.BLACK: 0
        }
        for color in total_value:
            for piece_type, value in self.piece_value.items():
                pieces = board.pieces(piece_type, color)
                for pos in pieces:
                    index = (-pos-1) if color else pos
                    total_value[color] += PIECE_SQUARE_TABLES[piece_type][index]
                    # logger.debug("%s PST %s @ %s - %s", color, piece_type, pos,  PIECE_SQUARE_TABLES[piece_type][index])
                total_value[color] += (len(pieces) * value)
        return total_value[side] - total_value[not side]
    

    def get_len_attacked_squares(self, board: chess.Board) -> dict[bool: int]:
        """Given a board, return difference in number of attacked squares for each side
        Positive means White is attacking more squares"""
        cached_ta = self.load_attacked_sq_cache(board)
        if cached_ta:
            # logger.debug("Returned cached value %s", cached_ta)
            return cached_ta
        total_attacks_diff = 0
        for color in (True, False): 
            for piece_type in range(2, 6):
                # exclude pawns and kings for optimization
                for square in board.pieces(piece_type, color):
                    total_attacks_diff += len(board.attacks(square)) * (1 if color else -1)
            # the following seems to take significantly longer
            # for attacker in chess.SquareSet(board.occupied_co[color]):
            #     attacks = board.attacks(attacker)
            #     total_attacks[color] += len(attacks)
        self.save_attacked_sq_cache(board, total_attacks_diff)
        return total_attacks_diff
    
    def get_value_attacked_pieces(self, board: chess.Board) -> dict[bool: int]:
        """Value of pieces which are under threat by the other side"""
        attacked_value = {
            True: 0,
            False: 0
        }
        for color in (True, False):
            for piece_type in range(2, 6):
                for square in board.pieces(piece_type, color):
                    if board.is_attacked_by(not color, square):
                        attacked_value[color] += self.piece_value[piece_type]
        return attacked_value
    

class ChessBot1999:
    def __init__(self, board = None, human: bool | None = None):
        self.board = board or chess.Board()
        self.move_count = 0
        self.human = human

    @staticmethod
    def print_position_info(board: chess.Board):
        print(f"{len(list(board.legal_moves))} legal moves")

    @staticmethod
    def make_random_move(board: chess.Board) -> chess.Move:
        move = choice(list(board.legal_moves))
        logger.debug(f"Selecting random move: {move}")
        return move

    def get_ai_move(self, search_depth = 3) -> chess.Move:
        """Make moves that immediately result in victory.
        Otherwise make a move that results in check.
        Otherwise make a move that results in continuing the game.
        Otherwise make a move that results in stalemate.
        """
        with Progress() as progress:
            me = MoveEvaluator1998(search_depth, progress)
            move = me.get_move(self.board, search_depth)
            me.db_cache_attacked_sq.dump()  # save attacked_squares cache
            logger.debug("Carefully selected move: %s", move)
            return move

    @staticmethod
    def human_chess_move(board: chess.Board) -> chess.Move:
        """Solicit a move from a human"""
        logger.info("Available legal moves: %s", ', '.join([str(x) for x in board.legal_moves]))

        def try_move(move) -> chess.Move | None:
            """Try to get a valid move from human input"""
            parsers = [board.parse_san, board.parse_uci]
            for parser in parsers:
                try:
                    m = parser(move)
                    return m
                except ValueError:
                    ...

        while True:
            human_input = input("Input your move: ")
            if human_input.lower().strip() == "takeback":
                return "takeback"
            elif human_input.lower().strip() == "cancel":
                raise Exception
            # san: standard braic notation
            # i.e. `e4` or `Nf6` pr `Qxf7`
            resulting_move = try_move(human_input)
            if resulting_move in board.legal_moves:
                return resulting_move

    def play_chess(self, move_threshold = 1000):
        logger.info(self.board)
        while list(self.board.legal_moves):
            if self.move_count > move_threshold:
                logger.warning("Move count (%s) exceeds move threshold (%s)", self.move_count, move_threshold)
                break
            self.move_count += 1
            logger.info(f"Move {self.move_count}; it is {'White' if self.board.turn else 'Black'}'s turn")
            # if self.board.turn:
            #     # white is dumb
            #     # move = make_random_move(self.board)
            #     move = human_chess_move(self.board)
            #     if move == "takeback":
            #         self.board.pop()
            #         self.board.pop()
            #         i-=2
            #         continue
            # else:
            move = self.get_ai_move()
            logger.info(f"Making move {move}")
            self.board.push(move)
            logger.info(self.board)
            chess.svg.board(self.board)
            if self.board.is_check():
                logger.warning("CHECK!")
            if self.board.is_insufficient_material():
                logger.info("Insufficient material to continue")
                break
        logger.info("Game over!")
        logger.info(f"Outcome: {self.board.outcome()}")
    
    def make_move(self):

        def move_selection():
            if self.human == self.board.turn:
                move = self.human_chess_move(self.board)
                if move == "takeback":
                    self.board.pop()
                    self.board.pop()
                    self.move_count -= 2
                    logger.info("Performed takeback!")
                    return
            else:
                move = self.get_ai_move()
            return move

        logger.info(self.board)
        logger.info("__________________")
        self.move_count += 1
        logger.info(f"Move {self.move_count}; it is {'white' if self.board.turn else 'black'}'s turn")
        move = move_selection()
        logger.info("%s moves %s", "White" if self.board.turn else "Black", move)
        self.board.push(move)
        if self.board.is_check():
            logger.warning("CHECK!")
        if self.board.is_insufficient_material():
            logger.info("Insufficient material to continue")
            return
        if not self.board.legal_moves:
            logger.info("Game over!")
            logger.info(f"Outcome: {self.board.outcome()}")
            return
        if self.human is not None and self.human != self.board.turn:
            # human just took their turn, start the AI's right away
            return self.make_move()

if __name__ == "__main__":
    cb = ChessBot1999(human=chess.WHITE)
    # import cProfile
    # cProfile.run('cb.play_chess(move_threshold=2)')
    while cb.board.legal_moves:
        cb.make_move()
    # games = {x: play_chess().outcome for x in range(5)}
    # print([g.winner for i, g in games.items()])
