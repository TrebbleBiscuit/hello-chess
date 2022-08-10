import chess
from random import choice, choices
import logging
from rich.logging import RichHandler
from rich.progress import Progress
import chess.svg
import sys
import pickledb
import json
from pathlib import Path
sys.path.append("./")
from hello_chess.evaluator import MoveEvaluator1998

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


class ChessBot1999:
    def __init__(self, board: chess.Board = None, human: bool | None = None, search_depth: int = 4, opening_book: Path | dict | None = Path("opening_book.json")):
        """
        Args:
            board (chess.Board, optional): _description_. Defaults to None.
            human (bool | None, optional): _description_. Defaults to None.
            search_depth (int, optional): _description_. Defaults to 4.
            opening_book (Path | dict | None, optional): Path to opening book json, or pass in a dict. Defaults to None.
        """

        self.board = board or chess.Board()
        self.move_count = 0
        self.human = human
        self._db_cache_mm = pickledb.load("mm_cache.db", False)
        self._db_cache_board_eval = pickledb.load("board_eval_cache.db", False)
        self.search_depth = search_depth
        if opening_book or self.board.fen() == chess.Board().fen():
            if isinstance(opening_book, dict):
                self.opening_book = opening_book
            elif isinstance(opening_book, Path):
                logger.debug("Loading opening book file %s; %s bytes...", opening_book, opening_book.stat().st_size)
                with open(opening_book, 'r') as f:
                    self.opening_book = json.load(f)
            else:
                self.opening_book = {}
        else:
            self.opening_book = {}

    @staticmethod
    def print_position_info(board: chess.Board):
        print(f"{len(list(board.legal_moves))} legal moves")

    @staticmethod
    def make_random_move(board: chess.Board) -> chess.Move:
        move = choice(list(board.legal_moves))
        logger.debug(f"Selecting random move: {move}")
        return move
    
    def _choose_from_opening_book_old(self):
        try:
            a = self.opening_book
            for m in self.board.move_stack:
                a = a[str(m)]
            if not a:
                logger.debug("Exhausted opening book")
                self.opening_book = {}
            else:
                move = chess.Move.from_uci(
                    # choice(
                    list(a.keys())[0]  # [0] instead of choice makes it deterministic
                    # )
                )
                logger.debug(f"Selected move from opening book: %s", move)
                return move
        except KeyError:
            logger.debug("Could not find next move %s in move stack: %s", m, self.board.move_stack)
            self.opening_book = {}
    
    def choose_from_opening_book(self):
        fen = self.board.fen()
        try:
            possible_moves = self.opening_book[fen]
            move = choices(
                population=list(possible_moves.keys()),
                weights=list(possible_moves.values()),
                k=1
            )[0]
            logger.info("Selected move from opening book: %s", move)
            return chess.Move.from_uci(move)
        except KeyError:
            self.opening_book = {}

    def get_ai_move(self) -> chess.Move:
        """Make moves that immediately result in victory.
        Otherwise make a move that results in check.
        Otherwise make a move that results in continuing the game.
        Otherwise make a move that results in stalemate.
        """
        # Try to use opening book first
        if self.opening_book:
            if len(self.board.move_stack) >= 10:
                logger.info("Not using opening book after 10 moves")
                self.opening_book = {}
            move = self.choose_from_opening_book()
            if move:
                return move
        with Progress() as progress:
            me = MoveEvaluator1998(self.search_depth, progress)
            me.db_cache_mm = self._db_cache_mm
            me.db_cache_board_eval = self._db_cache_board_eval

            move, value = me.iterative_search(self.board, self.board.turn, max_depth = self.search_depth)

            # value, move_list, chain = me.minimax(self.board, self.search_depth, -999999, 999999, self.board.turn)
            # move_list.sort(key=lambda x: -int(list(x.values())[0]))
            # logger.debug(move_list)
            # for value in assessment[0].values():
            #     ...  # just defining move and value of best move to use later
            # assert value == value2

            # logger.debug(me.board_evaluation)
            logger.debug(f"Trimmed {me.total_trimmed_moves} trees during evaluation")
            logger.info("\nCarefully selected move: %s with value %s \n", move, value)
            # logger.debug("me.db_cache_board_eval.dump()")
            # me.db_cache_board_eval.dump()
            if me.saved_to_db_cache_mm:
                logger.debug(f"Move evaluator saved to cache {me.saved_to_db_cache_mm} times")
                logger.debug("DUMPING CACHE, DO NOT INTERRUPT...")
                me.db_cache_mm.dump()
                logger.debug("Done dumping cache!")
            return move

    @staticmethod
    def human_chess_move(board: chess.Board) -> chess.Move:
        """Solicit a move from a human"""
        logger.info(
            "Available legal moves: %s", ", ".join([str(x) for x in board.legal_moves])
        )

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

    def play_chess(self, move_threshold=1000):
        logger.info(self.board)
        while list(self.board.legal_moves):
            if self.move_count >= move_threshold:
                logger.warning(
                    "Move count (%s) reached move threshold (%s)",
                    self.move_count,
                    move_threshold,
                )
                break
            self.move_count += 1
            logger.info(
                f"Move {self.move_count}; it is {'White' if self.board.turn else 'Black'}'s turn"
            )
            logger.info("Board fen: %s", self.board.fen())
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
                    logger.info(self.board)
                    return move_selection()
            else:
                move = self.get_ai_move()
            return move

        logger.info(self.board)
        logger.info("__________________")
        self.move_count += 1
        logger.info(
            f"Move {self.move_count}; it is {'white' if self.board.turn else 'black'}'s turn"
        )
        logger.info("Board fen: %s", self.board.fen())
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


SOME_FENS = [
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",   # e4 c5 Nf3
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",  # e4 c5 Nf3 d6
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", # e4 e5 Nf3 Nc6 
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",     # e4 e6 
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",      # d4
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",    # d4 Nf6 c4
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",    # d4 d5 c4
    "rnbqkb1r/pppppppp/5n2/8/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq c3 0 2",  # Nf3 Nf6 c4  (english)
]
from random import shuffle
shuffle(SOME_FENS)

def train():
    #for depth in [4, 6]:
    depth = 6
    for fen in SOME_FENS:
        logger.info(f"Preparing to evaluate moves at depth {depth} from: {fen}")
        cb = ChessBot1999(board=chess.Board(fen), search_depth=depth)
        for x in range(16):
            if cb.board.legal_moves:
                cb.make_move()
        logger.info(f"This game's outcome: {cb.board.outcome()}")
    
    cb = ChessBot1999(board=chess.Board(fen), search_depth=8)
    for x in range(8):
        if cb.board.legal_moves:
            cb.make_move()

# above 50k masters games https://www.365chess.com/opening.php
# COMMON_OPENINGS = {
#     "e4": {
#         "c5": {
#             "Nf3": {
#                 "d6": {
#                     "d4": {}
#                 },
#                 "e6": {
#                     "d4": {}
#                 },
#             }
#         },
#         "e5": {
#             "Nf3": {}
#         },
#         "e6": {
#             "d4", {}
#         },
#         "c6": {
#             "d4", {}
#         },
#     },
#     "d4": {
#         "Nf6": {
#             "c4": {
#                 "e6": {},
#                 "g6": {}
#             },
#             "Nf3": {},
#         },
#         "d5": {
#             "c4": {}
#         },
#     },
#     "Nf3": {
#         "Nf6": {},
#     },
#     "c4": {},
# }


if __name__ == "__main__":
    # train()
    cb = ChessBot1999(board=chess.Board(), human=chess.BLACK, search_depth=4, opening_book=Path("opening_book_20-10.json"))  # human=chess.BLACK
    while cb.board.legal_moves:
        cb.make_move()
    logger.info(cb.board.outcome())
    # # import cProfile
    # # cProfile.run('cb.play_chess(move_threshold=1)', sort="tottime")
    # for x in range(10):
    #     cb = ChessBot1999(human=None, search_depth=4)  # human=chess.BLACK
    #     for y in range(20):
    #         if cb.board.legal_moves:
    #             cb.make_move()
    #     if cb.board.outcome():
    #         logger.info(cb.board.outcome)
    # # games = {x: play_chess().outcome for x in range(5)}
    # # print([g.winner for i, g in games.items()])
