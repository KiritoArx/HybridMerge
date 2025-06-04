#include "ai.hpp"
#include <vector>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <iostream>

bool columnIsFull(const Board& board, int col) {
    return static_cast<int>(board[col].size()) >= MAX_HEIGHT;
}

Board simulateDrop(const Board& board, int value, int col) {
    Board newBoard = board;
    dropAndResolve(newBoard, value, col);
    return newBoard;
}

// New: Evaluate the result of a two-move sequence
int evaluateTwoMoveBoard(const Board& board, int scoreBefore, int scoreAfter, int highestTileBefore, int highestTileAfter) {
    int score = 0;
    int emptyCells = 0;
    int dangerPenalty = 0;

    // Heavily reward total points scored (prefers combos)
    score += (scoreAfter - scoreBefore) * 20;

    // Reward for highest tile made (encourages big merges)
    score += (highestTileAfter - highestTileBefore) * 400;

    // Reward empty space (keep options open)
    for (int col = 0; col < COLUMN_COUNT; ++col) {
        int h = static_cast<int>(board[col].size());
        if (h < MAX_HEIGHT) emptyCells += (MAX_HEIGHT - h);
        if (h >= MAX_HEIGHT - 1) dangerPenalty += 5000; // Avoid nearly full
        if (h >= MAX_HEIGHT) dangerPenalty += 20000;    // Dead column
    }
    score += emptyCells * 10;
    score -= dangerPenalty;

    return score;
}

// Utility to get board sum and highest tile
void getBoardStats(const Board& board, int& boardSum, int& highestTile) {
    boardSum = 0;
    highestTile = 0;
    for (const auto& col : board)
        for (int v : col) {
            boardSum += v;
            if (v > highestTile) highestTile = v;
        }
}

// Main AI: plan two moves ahead (current tile + preview tile)
int findBestMove(const Board& board, int currentTile, int previewTile, int searchDepth) {
    int bestMove = -1;
    int bestScore = std::numeric_limits<int>::min();

    // Pre-calculate stats for the current board
    int boardSumBefore, highestTileBefore;
    getBoardStats(board, boardSumBefore, highestTileBefore);

    // For each possible first move (current tile)
    for (int firstCol = 0; firstCol < COLUMN_COUNT; ++firstCol) {
        if (columnIsFull(board, firstCol)) continue;

        Board afterFirstMove = simulateDrop(board, currentTile, firstCol);

        // For each possible second move (preview tile)
        int bestSecondScore = std::numeric_limits<int>::min();
        for (int secondCol = 0; secondCol < COLUMN_COUNT; ++secondCol) {
            if (columnIsFull(afterFirstMove, secondCol)) continue;

            Board afterSecondMove = simulateDrop(afterFirstMove, previewTile, secondCol);

            int boardSumAfter, highestTileAfter;
            getBoardStats(afterSecondMove, boardSumAfter, highestTileAfter);

            // Score for this two-move sequence
            int seqScore = evaluateTwoMoveBoard(
                afterSecondMove, boardSumBefore, boardSumAfter, highestTileBefore, highestTileAfter);

            if (seqScore > bestSecondScore) {
                bestSecondScore = seqScore;
            }
        }

        // Prefer first moves that enable best two-move outcomes
        if (bestSecondScore > bestScore) {
            bestScore = bestSecondScore;
            bestMove = firstCol;
        }
    }
    // Fallback: If no good move, pick leftmost valid column
    if (bestMove == -1) {
        for (int col = 0; col < COLUMN_COUNT; ++col)
            if (!columnIsFull(board, col)) return col;
    }
    return bestMove;
}

void displayBoard(const Board& board, int markerCol, int markerRow) {
    for (int r = MAX_HEIGHT - 1; r >= 0; --r) {
        for (int c = 0; c < COLUMN_COUNT; ++c) {
            if (c == markerCol && r == markerRow) {
                std::cout << std::setw(6) << "?";
            }
            else if (r < static_cast<int>(board[c].size())) {
                std::cout << std::setw(6) << board[c][r];
            }
            else {
                std::cout << std::setw(6) << ".";
            }
        }
        std::cout << "\n";
    }
    std::cout << std::string(COLUMN_COUNT * 6, '-') << "\n";
    for (int c = 0; c < COLUMN_COUNT; ++c)
        std::cout << std::setw(6) << (c + 1);
    std::cout << "\n\n";
}
