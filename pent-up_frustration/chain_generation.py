import numpy as np
from numpy.typing import NDArray
from numba import njit

max_illegal_pattern_len = 4
useful_pattern_len = 2 + 7 - 1 + max_illegal_pattern_len
chain_begs = (np.array((0, 0)), np.array((0, 1)))


def gen_block_of_moves(forbidden: tuple[NDArray[int]],
                       stumps: NDArray[int],
                       n_in_block: int = 10_000
                       ) -> list[NDArray[int]]:
    '''
    Generates a list of "n_in_block" sets of "moves", each describing how one 
    chain of Pentagons will be laid out

    Each entry of the list cotains 16 "moves", describing where the next 
    pentagon should be placed comparing to the last one added (the first 
    pentagon is always at 0,0). Each move explains on which edge of the last 
    pentagon of the chain we will be placing the next
    
    Parameters:
        forbidden (tuple[NDArray[int]]): a set of moves we know not to be feasible
        
        stumps (NDArray[int]): sets of moves of len==7 which we have already
        tried, and know to be possible 
        
        n_in_block (optional) (int): number of moves to be placed in each blocks

    Returns:
        - list: len == n_in_block, where each item is a NDArray of ints of shape 16,
    '''
    gen = gen_valid_moves(forbidden, stumps.copy())
    db, i = [], 1

    for chain in gen:
        i += 1
        db.append(chain)
        if i == n_in_block:
            yield db
            db, i = [], 1
    yield db


@njit
def check_if_moves_invalid(moves: NDArray[int],
                           forbidden: tuple[NDArray[int]]
                           ) -> bool:

    '''
    checks if a set of moves contains moves that are known to somehow fold the
    chain back on itself - i.e. either by comparing against a tuple of 
    forbidden moves or by seeing whether the n+1 and the n-1 pentagons are on 
    the same side of the nth pentagon. 

    Parameters:
        moves (NDArray[int]): a ndarray of moves 
        forbidden: a tuple of forbidden moves of len < len(moves)
    Returns:
        bool: True if moves are invalid
    '''

    if ((4 - moves[:-1]) == moves[1:]).any():
        return True

    for y in forbidden:
        if seq_in_seq(x, y):
            return True

    return False


def gen_valid_moves(forbidden: tuple[NDArray[int]],
                    stumps: NDArray[int]
                    ) -> NDArray[int]:
    '''
    Generates a valid set of 16 moves (describing a chain of pentagons

    Paramters:
        forbidden (tuple[NDArray[int]]): set of short moves known to cause
        overlaps)

        stumps (NDArray[int]): set of moves known not to overlap.

    Returns
        NDArray[int]: an array of 16 moves (integers [0-4])
    '''
    
    last_chain_part = np.array([5] * useful_pattern_len)
    last_is_invalid = False
    for beg in chain_begs:
        for s1 in stumps:
            for s2 in stumps:
                chain = np.hstack((beg, s1, s2))
                chain_part = chain[:useful_pattern_len]
                if last_is_invalid:
                    if (chain_part == last_chain_part).all():
                        continue

                last_chain_part = chain_part
                if check_if_moves_invalid(chain_part, forbidden):
                    last_is_invalid = True
                    continue

                last_is_invalid = False
                yield chain


@njit
def seq_in_seq(big: NDArray[int], small: NDArray[int]) -> bool:
    '''Checks if small sequence is a subset of big sequence'''
    
    for x in range(len(big) - len(small)):
        for a, b in zip(big[x:], small):
            same = a == b
            if not same:
                break
        if same:
            return True
    return False