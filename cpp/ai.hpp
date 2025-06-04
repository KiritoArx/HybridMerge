#pragma once
#include "merge.hpp"
#include <vector>

// Find the best move (column) given the board, current tile, and preview tile.
// searchDepth is ignored (always 2), but included for compatibility.
int findBestMove(const Board& board, int currentTile, int previewTile, int searchDepth = 2);


// Pretty-print the board; markerCol and markerRow default to -1 (no marker)
void displayBoard(const Board& board, int markerCol = -1, int markerRow = -1);

