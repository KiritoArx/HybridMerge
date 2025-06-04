#ifndef GAME_DEFS_HPP
#define GAME_DEFS_HPP

#include <vector>
#include <deque> // Included for potential use, Board itself uses vector<vector>

// Define fundamental types for clarity
using Tile = int;
using Column = std::vector<Tile>;  // Each column is a vector of tiles
using Board = std::vector<Column>;  // The board is a vector of columns

// === Game Configuration ===
const int COLUMN_COUNT = 5; // Number of columns on the game board
const int MAX_HEIGHT = 6;   // Maximum number of tiles allowed in a column
// ==========================

#endif // GAME_DEFS_HPP
