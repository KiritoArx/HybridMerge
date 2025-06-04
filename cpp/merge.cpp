#include "merge.hpp"
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>

// Pretty-printer (tight ASCII) - same as before
void printBoard(const Board& board) {
    for (int row = MAX_HEIGHT - 1; row >= 0; --row) {
        for (int col = 0; col < COLUMN_COUNT; ++col) {
            if (col < static_cast<int>(board.size()) && row < static_cast<int>(board[col].size()))
                std::cout << std::setw(6) << board[col][row];
            else
                std::cout << std::setw(6) << '.';
        }
        std::cout << '\n';
    }
    std::cout << std::string(COLUMN_COUNT * 6, '-') << '\n';
    for (int col = 0; col < COLUMN_COUNT; ++col)
        std::cout << std::setw(6) << ("C" + std::to_string(col));
    std::cout << "\n\n";
}

// Drop-and-resolve
int dropAndResolve(Board& board, Tile value, int col) {
    std::cout << "[MERGE_DEBUG] Entering dropAndResolve. Tile=" << value << ", Col=" << col << std::endl;
    // std::cout << "[MERGE_DEBUG] Board BEFORE drop:" << std::endl;
    // printBoard(board); // Can be very verbose if called often

    if (col < 0 || col >= COLUMN_COUNT || (board.size() > static_cast<size_t>(col) && board[col].size() >= MAX_HEIGHT) ) {
         std::cout << "[MERGE_DEBUG] Invalid move or column full. Col=" << col << ", Value=" << value << std::endl;
        return 0; 
    }
    // Ensure board has enough columns if it was dynamically sized (though ours is fixed by COLUMN_COUNT at construction)
    if (board.size() <= static_cast<size_t>(col)) {
        // This case should ideally not happen if Board is initialized with COLUMN_COUNT columns.
        // board.resize(col + 1); // Or handle as an error
        std::cerr << "[MERGE_DEBUG] Error: Column index out of bounds for board.size(). col=" << col << ", board.size()=" << board.size() << std::endl;
        return 0; // Cannot proceed
    }


    board[col].push_back(value);
    // int R = static_cast<int>(board[col].size()) - 1; // Not directly used in the refined logic below

    int score_from_merges = 0;
    bool changed_in_iteration = true;

    // Max iterations to prevent infinite loops in case of unforeseen issues
    int max_resolve_loops = (MAX_HEIGHT * COLUMN_COUNT) * 2 + 5; 
    int resolve_loop_count = 0;

    while(changed_in_iteration && resolve_loop_count < max_resolve_loops){
        resolve_loop_count++;
        changed_in_iteration = false;
        // std::cout << "[MERGE_DEBUG] Resolve Loop Iteration: " << resolve_loop_count << std::endl;

        // 1. Apply Gravity to all columns
        for(int c_idx = 0; c_idx < COLUMN_COUNT; ++c_idx) {
            if (board[c_idx].empty()) continue;
            std::vector<Tile> new_col;
            for(Tile t : board[c_idx]){
                if(t != 0) new_col.push_back(t); // Assuming 0 means an empty space after a merge
            }
            if(new_col.size() != board[c_idx].size()){
                changed_in_iteration = true;
                board[c_idx] = new_col;
                // std::cout << "[MERGE_DEBUG] Gravity applied to col " << c_idx << std::endl;
            }
        }

        // 2. Process Merges (Iterate multiple times if one merge enables another)
        // This simplified merge pass might need more sophistication for complex chain reactions.
        // For now, one pass checking all tiles.
        for (int c = 0; c < COLUMN_COUNT; ++c) {
            for (int r = 0; r < static_cast<int>(board[c].size()); ++r) {
                if (board[c][r] == 0) continue; // Skip already merged tiles
                Tile v = board[c][r];

                // Vertical-down merge (Primary vertical merge)
                if (r > 0 && board[c][r-1] == v) {
                    // std::cout << "[MERGE_DEBUG] Vertical merge at (" << c << "," << r << ") with (" << c << "," << r-1 << ")" << std::endl;
                    board[c][r-1] *= 2;
                    score_from_merges += board[c][r-1];
                    board[c].erase(board[c].begin() + r);
                    changed_in_iteration = true;
                    r--; // Re-check the new merged tile at r-1 in the next iteration (or immediately if logic allows)
                    continue; 
                }

                // Horizontal merges (check right neighbor only to avoid double checks/merges)
                if (c < COLUMN_COUNT - 1) { // If not the rightmost column
                    // Ensure row r is valid for neighbor column nc
                    if (r < static_cast<int>(board[c+1].size()) && board[c+1][r] == v) {
                        // std::cout << "[MERGE_DEBUG] Horizontal merge at (" << c << "," << r << ") with (" << c+1 << "," << r << ")" << std::endl;
                        // Merge into the left tile (c,r)
                        board[c][r] *= 2;
                        score_from_merges += board[c][r];
                        board[c+1].erase(board[c+1].begin() + r);
                        changed_in_iteration = true;
                        // No need to r-- here as the current tile (c,r) changed.
                        // The column c+1 changed, gravity will handle it.
                    }
                }
            }
        }
        if (resolve_loop_count >= max_resolve_loops) {
            std::cout << "[MERGE_DEBUG] Warning: Exceeded max_resolve_loops. Breaking." << std::endl;
        }
    }
    // std::cout << "[MERGE_DEBUG] Exiting dropAndResolve. Score=" << score_from_merges << std::endl;
    // printBoard(board);
    return score_from_merges;
}

// gameOver, boardSum, columnFull, maxTile - same as before
bool gameOver(const Board& board) {
    for (int c = 0; c < COLUMN_COUNT; ++c) {
        if (board[c].size() >= MAX_HEIGHT) {
            return true;
        }
    }
    return false;
}

int boardSum(const Board& board) {
    int sum = 0;
    for (int c = 0; c < COLUMN_COUNT; ++c) {
        for (Tile tile_val : board[c]) {
            sum += tile_val;
        }
    }
    return sum;
}

bool columnFull(const Board& board, int col) {
    if (col < 0 || col >= COLUMN_COUNT) return true;
    return board[col].size() >= MAX_HEIGHT;
}

Tile maxTile(const Board& board) {
    Tile max_val = 0; 
    bool first_tile_found = false;
    for (int c = 0; c < COLUMN_COUNT; ++c) {
        for (Tile tile_val : board[c]) {
            if (!first_tile_found) {
                max_val = tile_val;
                first_tile_found = true;
            } else if (tile_val > max_val) {
                max_val = tile_val;
            }
        }
    }
    return max_val;
}
