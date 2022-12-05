import numpy as np
from itertools import product
from joblib import Parallel, delayed

from pentagon_and_chain import Chain
from chain_generation import gen_block_of_moves


def get_min_dist_from_chain(
        moves: np.ndarray[int]) -> tuple[float | None, int | None]:
    '''
    takes in a set of moves (which describes the chain of 17 pentagons) and returns the minimum nonzero distance from any of those pentagons
    Parameters:
        moves (np.ndarray[int]): an array of 17 integers 0-4, explaining how the chain of pentagons should be laid out
    Returns
        tuple[float, int]: where the float represents the minimum distance (if chain was valid, otherwise it is none). If the distance is none, the int represents the step at which it overlapped. 
    '''

    c = Chain()
    success, i = c.from_summary(moves)
    if success:
        dist = c.min_nonzero_distance()
        return dist, None
    else:
        return None, i


def get_min_dist_from_batch(
        batch: list[np.ndarray[int]]) -> tuple[np.ndarray[int], float]:
    '''
    applies the get_min_dist_from_batch function to a given batch. 
    To be a little smarter, we keep track of which chains were overlapping, and if a new chain contains the overlapping pattern we don't even look at it

    Parameters:
        batch (list[np.ndarray[int]]): a list of moves, each represnting a plausible layout of 17 pentagons
    Returns:
       tuple[np.ndarray[int], float]: respectively the moves that generated the set of pentagons with the minimum distance, and that distance 
    '''

    best = 0, 10
    bad = None
    for moves in batch:
        if bad is not None:
            if all(moves[:len(bad)] == bad):
                continue

        dist, last_step = get_min_dist_from_chain(moves)
        if last_step is not None:
            bad = moves[:last_step + 1]

        if dist is not None:
            bad = None
            if dist < best[1]:
                best = moves, dist

    return best


def quick_exploration():
    '''
    Looking at all the combinations of 7 moves, to quickly get a sense of what moves are valid and what are invalid
    Returns:
        forbidden (tuple): tuple of all short sequences that were found to cause an overlap
        stumps (np.ndarray): nx7 array of moves that were found not to cause an overlap 
    '''
    stumps = []
    forbidden = set()
    for chain_moves in product(range(4), repeat=7):
        chain = Chain()
        for step in chain_moves:
            success = chain.make_new(free_edge=step)
            if not success:
                bad_move = chain.summary[chain.error['overlaps_with']:]
                actual_move = [
                    x for x in range(5) if x != chain.pentagons[-1].busy_edge
                ][step]
                bad_move += tuple([actual_move])
                forbidden.add(bad_move)
                break

        if success:
            stumps.append(chain.summary)

    stumps = np.array(stumps)
    forbidden = sorted(forbidden, key=len)
    forbidden = tuple(np.array(x) for x in forbidden if len(x) <= 4)
    return forbidden, stumps


# Step 1. Exploration of a behaviour of 8 pentagons
invalid_moves, valid_moves = quick_exploration()

# Step 2. Computing minimum distances on all chains worth an attempt
out = Parallel(n_jobs=10, verbose=10)(
    delayed(get_min_dist_from_batch)(batch)
    for batch in gen_block_of_moves(invalid_moves, valid_moves))

moves = [x[0] for x in out]
dists = [x[1] for x in out]
solution = min(dists)  # done!

# Step 3. Saving Example solutions and graphs
mm = np.argmin([round(x, 12) for x in dists])

possibles = [
    moves for moves, dist in out if round(dist, 7) == round(dists[mm], 7)
]

# saving solutions
for i, x in enumerate(possibles):
    c = Chain()
    _ = c.from_summary(x)
    ax = c.plot(path=f'solution/plots/example {i}.png')

with open('solution/example solutions.csv', 'w') as f:
    for solution in possibles:
        solution = [str(x) for x in solution]
        _ = f.write(", ".join(solution))
        _ = f.write("\n")