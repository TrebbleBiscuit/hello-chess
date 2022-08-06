import chess
from random import choice
import logging
from rich.logging import RichHandler
from rich.progress import Progress
import chess.svg
from hello_chess.evaluator import MoveEvaluator1998

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


class ChessBot1999:
    def __init__(self, board=None, human: bool | None = None):
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

    def get_ai_move(self, search_depth=3) -> chess.Move:
        """Make moves that immediately result in victory.
        Otherwise make a move that results in check.
        Otherwise make a move that results in continuing the game.
        Otherwise make a move that results in stalemate.
        """
        with Progress() as progress:
            me = MoveEvaluator1998(search_depth, progress)
            value, move = me.minimax(None, self.board, search_depth)
            logger.debug("me.db_cache_board_eval.dump()")
            me.db_cache_board_eval.dump()
            logger.debug(me.board_evaluation)
            logger.debug("Carefully selected move: %s with value %s", move, value)
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
        logger.info(
            f"Move {self.move_count}; it is {'white' if self.board.turn else 'black'}'s turn"
        )
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
    cb = ChessBot1999(human=chess.BLACK)
    # import cProfile
    # cProfile.run('cb.play_chess(move_threshold=1)', sort="tottime")
    # while cb.board.legal_moves:
    cb.make_move()
    # games = {x: play_chess().outcome for x in range(5)}
    # print([g.winner for i, g in games.items()])
