import chess
from random import choice, random
import logging
from rich.logging import RichHandler
from rich.progress import Progress
import chess.svg

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

def print_position_info(board: chess.Board):
    print(f"{len(list(board.legal_moves))} legal moves")

def make_random_move(board: chess.Board) -> chess.Move:
    move = choice(list(board.legal_moves))
    logger.debug(f"Selecting random move: {move}")
    return move

def _make_smarter_random_move(board: chess.Board, search_depth, progress: Progress, progress_threshold: int) -> chess.Move:
    """Don't call this directly, use the wrapper without the preceeding underscore instead"""
    this_side = chess.WHITE if board.turn else chess.BLACK
    # move_eval = {}
    tie_moves = []
    continuing_moves = []  # these do not end the game

    checking_moves = []  # these cause check

    for move in board.legal_moves:
        # let's imagine we did make this move
        board.push(move)
        if board.outcome():
            # this move ends the game
            if board.outcome().winner == this_side:  # victory
                board.pop()  # undo this move so we can immediately make it again
                return move
            if not board.outcome().winner:
                tie_moves.append(move)
            # elif board.outcome().winner != this_side:
            #     ...  # avoid losses
        else:
            if board.is_check():
                # both checking and continuous
                checking_moves.append(move)
            continuing_moves.append(move)
        board.pop()  # undo move
    # now we've categorized various moves
    # how do we determine value of moves within categories?
    # want to look one move ahead - can opponent immediately win?
    # logger.debug({
    #     "tie_moves": len(tie_moves),
    #     "checking_moves": len(checking_moves),
    #     "continuing_moves": len(continuing_moves)
    # })
    if not continuing_moves:
        return tie_moves[0]
    # return choice(continuing_moves)
    # logger.debug(f"Evaluating {len(continuing_moves)} moves for {'White' if this_side else 'Black'}")
    if search_depth >= progress_threshold:
        search_task = progress.add_task(f"Searching {len(continuing_moves)} moves at depth {search_depth}", total=len(continuing_moves))
    move_eval = evaluate_potential_moves(continuing_moves, search_depth, progress)
    return max(move_eval, key=move_eval.get)  # return the best move

def evaluate_potential_moves(potential_moves: list[chess.Move], search_depth: int, progress) -> dict[chess.Move: float]:
    """Given a selection of moves, score them"""
    this_side = chess.WHITE if board.turn else chess.BLACK
    move_eval = {}  # map of moves to evaluated score
    for move in potential_moves:
        # let's imagine we made this move
        board.push(move)
        if search_depth == 0:
            score = evaluate_material(board, this_side)
        elif search_depth > 0:
            # what does our opponent's reponse look like?
            # we want our "score" for this position to reflect
            # the opponent's best move next turn
            o_move = _make_smarter_random_move(board, search_depth - 1, progress, progress_threshold)
            # if that happened
            board.push(o_move)
            # what would the result be?
            score = evaluate_material(board, this_side)
            board.pop()
        # moves that cause check are a little better
        if move in checking_moves:
            score += 0.15
        # attacking more squares is good; nudge score based on
        # total number of attacked squares to encourage development
        atks = get_len_attacked_squares(board)
        diff_squares_attacked = atks[this_side] - atks[not this_side]
        score += diff_squares_attacked * 0.02
        # add a v small random amount to score so that when
        # we use max() on a bunch of options with the same
        # value it selects one randomly
        score += random() * 0.1
        move_eval[move] = score
        board.pop()
        if search_depth >= progress_threshold:
            progress.update(search_task, advance=1)
    return move_eval
    # logger.debug(move_eval)
    

def make_smarter_random_move(board: chess.Board, search_depth = 3) -> chess.Move:
    """Make moves that immediately result in victory.
    Otherwise make a move that results in check.
    Otherwise make a move that results in continuing the game.
    Otherwise make a move that results in stalemate.
    """
    with Progress() as progress:
        move = _make_smarter_random_move(board, search_depth, progress, search_depth)
        logger.debug("Carefully selected move: %s", move)
        return move

def evaluate_material(board: chess.Board, side: bool) -> float:
    """Board evaluation is equal to material value minus opponent's"""
    if board.outcome():
        if not board.outcome().winner:  # tie
            return 0
        if board.outcome().winner == side:
            return 200
        else:
            return -200
    piece_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    total_value = {
        chess.WHITE: 0,
        chess.BLACK: 0
    }
    for color in total_value:
        for piece, value in piece_value.items():
            total_value[color] += len(board.pieces(piece, color)) * value
    return total_value[side] - total_value[not side]

def get_len_attacked_squares(board: chess.Board) -> dict[bool: int]:
    """Given a board, return number of attacked squares for each side"""
    total_attacks = {
        chess.WHITE: 0,
        chess.BLACK: 0
    }
    for color in [chess.WHITE, chess.BLACK]: 
        for piece_type in range(1, 6):
            for square in board.pieces(piece_type, color):
                attacks = board.attacks(square)
                total_attacks[color] += len(attacks)
    return total_attacks

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


def play_chess():
    board = chess.Board()
    i = 0
    logger.info(board)
    while list(board.legal_moves):
        if i > 1000:
            logger.warning("Breaking after 1000 moves")
            break
        i += 1
        logger.info(f"Move {i}; it is {'white' if board.turn else 'black'}'s turn")
        # if board.turn:
        #     # white is dumb
        #     # move = make_random_move(board)
        #     move = human_chess_move(board)
        #     if move == "takeback":
        #         board.pop()
        #         board.pop()
        #         i-=2
        #         continue
        # else:
        move = make_smarter_random_move(board)
        logger.info(f"Making move {move}")
        board.push(move)
        logger.info(board)
        chess.svg.board(board)
        if board.is_check():
            logger.warning("CHECK!")
        if board.is_insufficient_material():
            logger.info("Insufficient material to continue")
            break
    logger.info("Game over!")
    logger.info(f"Outcome: {board.outcome()}")
    return board

if __name__ == "__main__":
    board = play_chess()
    # games = {x: play_chess().outcome for x in range(5)}
    # print([g.winner for i, g in games.items()])
