import chess
from random import choice
board = chess.Board()
while not board.outcome():
    move = choice(list(board.legal_moves))
    board.push(move)