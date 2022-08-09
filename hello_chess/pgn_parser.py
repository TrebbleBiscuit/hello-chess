import chess.pgn
import json
from pathlib import Path
from copy import deepcopy
from rich.progress import track




def get_sequences_from_pgn(pgn_path: Path, sequences: dict = None) -> dict[str, int]:
    """Generates move sequences from PGN

    Args:
        pgn_path (Path): _description_
        sequences (dict): existing dictionary of sequences to merge with 

    Returns:
        dict[str, int]: Key is a string of the first ten moves (e.g. "e2e4,d7d5"), value is number of occurances
    """
    SEQUENCE_LENGTH = 20
    MIN_GAME_LENGTH = 40

    pgn = open(pgn_path)
    if sequences:
        all_seq = sequences
    else:
        all_seq = {}
    
    i = 1
    while True:  # keep reading through games until there aren't any left
        game = chess.pgn.read_game(pgn)
        if game is None:
            print(f"Finished processing {pgn_path} - {i} total games")
            break
        if i % 100000 == 0: print(f"Processed {i} games")
        mainline = list(game.mainline_moves())
        i += 1
        if len(mainline) < MIN_GAME_LENGTH:
            continue  # only look at games that were >= 20 moves long
        str_mov = ",".join(str(x) for x in mainline[0:SEQUENCE_LENGTH])
        if str_mov not in all_seq:
            all_seq[str_mov] = 1
        else:
            all_seq[str_mov] += 1
    return all_seq

# all_sequences = {}
# # pgn_file = Path("data/lichess_elite_2022-06.pgn")
# for pgn_file in Path("H:\Bulk Data\Lichess Elite Database").glob("*.pgn"):
#     all_sequences = get_sequences_from_pgn(pgn_file, all_sequences)

# # list of all unique move sequences
# unique_sequences = {}
# for i, seq in enumerate(move_sequences):
#     if len(seq) < 10:
#         continue  # ignore sequences less than 10 moves long
#     str_mov = ",".join(str(x) for x in seq)  # list of stringified moves
#     if str_mov not in unique_sequences:
#         unique_sequences[str_mov] = 1
#     else:
#         unique_sequences[str_mov] += 1


def prune_sequences(unique_sequences, freq_threshold):
    new_sequences = deepcopy(unique_sequences)
    n_removed = 0
    for mv, freq in unique_sequences.items():
        if freq < freq_threshold:
            n_removed += 1
            new_sequences.pop(mv)
    print(f"Removed {n_removed}/{len(unique_sequences)}")
    return new_sequences

# new_sequences = prune_sequences(unique_sequences, freq_threshold = 5)
# len(new_sequences)

def make_move_book(move_sequences) -> dict:
    move_book = {}
    total_qty = 0
    for seq, qty in move_sequences.items():
        total_qty += qty
        board = chess.Board()
        for move in seq.split(","):
            fen = board.fen()
            try:
                move_book[fen]
                try:
                    move_book[fen][move] += qty
                except KeyError:
                    move_book[fen][move] = qty
            except KeyError:
                move_book[fen] = {move: qty}
            board.push(chess.Move.from_uci(move))
    print(f"Made move book from first {len(seq)} moves of {total_qty} games")
    return move_book

    # def add_key_to_dict(mb_dict, key):
    #     try:
    #         mb_dict[key]
    #     except KeyError:
    #         mb_dict[key] = {}
    #     return mb_dict[key]

    # move_book = {}
    # for seq in move_sequences:
    #     a = move_book
    #     for move in seq.split(","):
    #         a = add_key_to_dict(a, move)
    # return move_book


# alt_form = {
#     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": {
#         "e2e4": 4241,
#         "d2d4": 1618,
#     }
# }

# with open('opening_book.json', 'w') as f:
#     json.dump(nmc, f)

# with open('opening_book.json', 'r') as f:
#     opening_book = json.load(f)

if __name__ == "__main__":
    import time
    start_time = time.time()
    all_sequences = {}
    # all_pgn_files = [Path("data/test_pgn.pgn")]
    all_pgn_files = list(Path("H:\Bulk Data\Lichess Elite Database").glob("*.pgn"))
    for pgn_file in track(all_pgn_files, description="Analyzing games from PGN files"):
        print(f"Now analyzing {pgn_file}")
        all_sequences = get_sequences_from_pgn(pgn_file, all_sequences)
        print("Dumping intermediate data file to raw_inermediate_20move_data.json")
        with open('raw_inermediate_20move_data.json', 'w') as f:
            json.dump(all_sequences, f)

    print("Dumping to raw_20move_data.json")
    with open('raw_20move_data.json', 'w') as f:
        json.dump(all_sequences, f)
    end_time = time.time()
    print(f"Time: {start_time - end_time}")

    pruned_sequences = prune_sequences(all_sequences, freq_threshold = 5)

    move_book = make_move_book(pruned_sequences)
    print("Dumping to move_book.json")
    with open('move_book.json', 'w') as f:
        json.dump(move_book, f)
