# SudokuColoring
Solving a sudoku puzzle using graph theory and various coloring algorithms


#How to use:
Change numColors to your grid size. The grid size should be a perfect square. For example, 4, 9, 16, 25 works. The higher up, the more complex and longer compute times.
Change solved to the initial state of your board. The top right square starts at zero. For example, if the initial state has a 1 in the first square, add (0,1) to the array. Repeat for all squares.
After those changes, run everything and a graph and completed board will be created. Note that it will run both algorithms, and the greedy will most likely fail


#Inspiration:
I was inspired by a youtube video from Polylog, discussing Sudoku and how it relates to graph theory.
I did not realize it was possible to solve using coloring algorithms, so I decided to give it a try.

I constructed the graph using networkx as a framework. In doing so, I added all appropriate edges to validate any puzzle.
For example, all nodes will be adjacent to all nodes in the same row, column, and box.

Using this, I used a basic greedy algorithm which fails quite often. It works on small puzzles, or easy ones.
I also created an algorithms inspired by Wave Function Collapsed (WFC)
WFC is normally used in a generative context, but adding back tracking allows it to attempt to solve the puzzle.
It follows a normal WFC structure, by defining domains and randomly assigning a value from the most constrained tile.
From there, it does checks for any immediate issues, then continues to the next one.
If at any point it hits a road block, it will remember its steps and try something new.

