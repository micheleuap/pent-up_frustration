# Pent-up_frustration
## Intro
This is my (accepted) solution to the [Nov 2022 Jane Street Problem](https://www.janestreet.com/puzzles/pent-up-frustration-2-index/).

This problem has no (known!) analytical solution - so what I do here is try to find a smart way to try plausible combinations. 

While there are 5^16 ways of creating chains of pentagons, quite a lot of them are obviously not worth trying. In particular:
- at the end of any chain, there are only 4 useful sides of a pentagon on which we can attach another one, as one is already taken - this brings us down to 4^16 combinations
- there are only two sets of 3 pentagons worth exploring, all others are just the same, but turned/flipped around. This brings us down to 2*4^14 combinations
- if we explore how the problem works out on a short chain of pentagons, we can learn what combinations do and do not work. A longer chain can be then built by attaching working chains to each other, and checking that their joining did not result in any sequence that is known to result in an overlap

This way we no longer have to try billions of combinations, but just a few million!

## House-Keeping
As a bit of house-keeping, each way of laying out the pentagons is represented as of sequences of moves, where each move is a number 0-4 that signals on which side of the last pentagon we will be attaching the next. In this way, to represent 17 pentagons we need 16 moves (the 17th pentagon is at 0,0)

## Strategy
I divided the excercise in three parts:
1. Exploration - looking at all the combinations of 7 moves, to quickly get a sense of what moves are valid and what are invalid. 
2. Simulation - where we try out all the plausible combinations of 16 moves. This is in itself divided in two:
    1. Chain Generation - the sets of 16 moves are generated as one of the two possible set of two initial moves (see point 2. in the intro), followed by some combination of the 
    2. Computing the minimum distance between pentagons - this is just a bit of high school maths
3. Making plots and saving results

## Code Structure
- main.py: contains the layout of the strategy above
- pentagon and chain: class definitions for a pentagon, and a chain of pentagons, as well as their useful methods - such as computing where the edges and vertices of pentagons are, and whether pentagons overlap, how distant the are etc.
- chain generation: definition of the generators of the sets of 16 moves, which describe how a chain of pentagons will need to be created

## Speeding up the process
To have this running in a decent time, without wasting a ton of my time, I optimised this in two ways:
1. The heavier functions are compiled JIT using numba
2. Section 2.2 of the strategy is ran in parallel, given that it is embarassingly parallel. This is done by creating batches of a few thousand sets of moves, having each worker compute the minimum distance from its batch. The absolute minimum difference is of course the minimum of the miniums from all batches

This now runs in under 4h on my laptop, so I've stopped here, but we could do significantly better.  
