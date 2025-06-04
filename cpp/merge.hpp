#ifndef MERGE_HPP
#define MERGE_HPP

#include "game_defs.hpp" // For Board, Tile, COLUMN_COUNT, MAX_HEIGHT

// Pretty-printer for the game board
void printBoard(const Board& board);

// Drops a tile into a column and resolves any resulting merges
// Returns the score obtained from merges in this step.
int dropAndResolve(Board& board, Tile value, int col);

// Checks if the game is over (e.g., a column is full)
bool gameOver(const Board& board);

// --- Utility functions often needed by AI or game logic ---

// Calculates the sum of all tile values on the board
int boardSum(const Board& board);

// Checks if a specific column is full
bool columnFull(const Board& board, int col);

// Finds the highest tile value currently on the board
Tile maxTile(const Board& board);

#endif // MERGE_HPP
