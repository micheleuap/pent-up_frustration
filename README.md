# Pent-up frustration 2
## Intro
This is my (accepted!) solution to the [Nov 2022 Jane Street Problem](https://www.janestreet.com/puzzles/pent-up-frustration-2-index/).

I tried a search-based approach, wich judging by the official solutions might have been the only way. 

## Strategy
In broad strokes, I first tried to reduce as much as possible the search space, and then looked for a solution within it. 

I divided the excercise in three parts

### 1. A priori reduction of the search space
While there are $5^{16}$ ways of creating chains of pentagons, quite a lot of them are obviously not worth trying. In particular:
- at the end of any chain, there are only 4 useful sides of a pentagon on which we can attach another one, as one is already taken. This brings us down to $4^{16}$ combinations
- there are only 2 sets of valid three pentagons chains, all others are just the same, but turned/flipped around. This brings us down to $2*4^{14}$ combinations

<img src="assets/min_chain_0.png" alt="Example 1" width="300"/>  <img src="assets/min_chain_1.png" alt="Example 2" width="300"/>

### 2. Breaking down the problem, and reducing further the complexity
Because there are only 2 useful chains of 3 pentagons worth investigating, I broke down a chain of 17 as a a chain-of-chains - i.e. chaining some chain of 3 pentagons, to some chain of 7 to some other chain of 7. 
Commputing all valid chains of 7 is fast, and yields a small subset of all chains of 7, further narrowing down the search space. 

### 3. Looking for a solution
Now that we only have a few million solutions to try, we can just walk through them all. I used some numba and some trivial parallelization to speed up the process.  This is in itself divided in two:
1. Chain Generation - the sets of 16 moves are generated as one of the two possible set of two initial moves (see point 2. in the intro), followed by some combination of two of the valid sets of 7 moves (2+7+7 = 16)
2. Computing the minimum distance between pentagons - this is just a bit of high school maths

This now relatively quickly on my laptop, so I've stopped here, but we could probably optimize this a fair bit more.  

# Example solutions 
A few are stored in the solution folder, but here is one that I think looks nice. Of course, the minimum distance is the one between pentagons 0 and 16

![solution](<assets/example 0.png>)

### Project Structure

```bash
📦pent-up-frustration
 ┣ 📂assets                  # Chain images for reacme.md
 ┣ 📂pent-up-frustration     # Contains all module scripts
  ┣ 📜pentagon_and_chain.py  # Contains class definitions for pentagons and chains, plotting and distance utils
  ┣ 📜chain_generation.py    # Functions for chain generation and detection of overlaps
  ┗ 📜main.py                # Main entrypoint
 ┗ 📜README.md               # This readme
```
