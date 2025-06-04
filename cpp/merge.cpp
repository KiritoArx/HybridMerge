#include "merge.hpp" // Should include game_defs.hpp for Board, Tile, COLUMN_COUNT, MAX_HEIGHT
#include <deque>
#include <iomanip>
#include <iostream> // Keep for printBoard, but remove from dropAndResolve
#include <string>
#include <numeric>   
#include <algorithm> 

// Pretty-printer (tight ASCII) - This is fine to keep for utility
void printBoard(const Board& board) {
    for (int row = MAX_HEIGHT - 1; row >= 0; --row) {
        for (int col = 0; col < COLUMN_COUNT; ++col) {
            // Ensure column c exists before trying to access its elements
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
// Returns score from merges in this step
int dropAndResolve(Board& board, Tile value, int col) {
    // Removed: std::cout << "[MERGE_DEBUG] Entering dropAndResolve. Tile=" << value << ", Col=" << col << std::endl;

    if (col < 0 || col >= COLUMN_COUNT || (board.size() > static_cast<size_t>(col) && board[col].size() >= MAX_HEIGHT) ) {
        // Removed: std::cout << "[MERGE_DEBUG] Invalid move or column full. Col=" << col << ", Value=" << value << std::endl;
        return 0; 
    }
    if (board.size() <= static_cast<size_t>(col)) {
        // This case should ideally not happen if Board is initialized with COLUMN_COUNT columns.
        // Removed: std::cerr << "[MERGE_DEBUG] Error: Column index out of bounds for board.size(). col=" << col << ", board.size()=" << board.size() << std::endl;
        return 0; 
    }

    board[col].push_back(value);
    int score_from_merges = 0;
    bool changed_in_iteration = true;
    int max_resolve_loops = (MAX_HEIGHT * COLUMN_COUNT) * 2 + 5; 
    int resolve_loop_count = 0;

    while(changed_in_iteration && resolve_loop_count < max_resolve_loops){
        resolve_loop_count++;
        changed_in_iteration = false;

        // 1. Apply Gravity to all columns
        for(int c_idx = 0; c_idx < COLUMN_COUNT; ++c_idx) {
            if (board[c_idx].empty()) continue;
            std::vector<Tile> new_col;
            new_col.reserve(board[c_idx].size()); // Pre-allocate memory
            for(Tile t : board[c_idx]){
                if(t != 0) new_col.push_back(t); 
            }
            if(new_col.size() != board[c_idx].size()){
                changed_in_iteration = true;
                board[c_idx] = new_col;
            }
        }

        // 2. Process Merges
        for (int c = 0; c < COLUMN_COUNT; ++c) {
            for (int r = 0; r < static_cast<int>(board[c].size()); ++r) { // Iterate up to current dynamic size
                if (board[c][r] == 0) continue; 
                Tile v = board[c][r];

                // Vertical-down merge
                if (r > 0 && board[c][r-1] == v) {
                    board[c][r-1] *= 2;
                    score_from_merges += board[c][r-1];
                    board[c].erase(board[c].begin() + r);
                    changed_in_iteration = true;
                    r--; // Re-check the new merged tile at r-1 in this column's pass
                    continue; 
                }

                // Horizontal merges (check right neighbor only)
                if (c < COLUMN_COUNT - 1) { 
                    if (r < static_cast<int>(board[c+1].size()) && board[c+1][r] == v) {
                        board[c][r] *= 2;
                        score_from_merges += board[c][r];
                        board[c+1].erase(board[c+1].begin() + r);
                        changed_in_iteration = true;
                        // After a horizontal merge, the current tile (c,r) has changed.
                        // The column c+1 also changed, gravity will handle empty spaces below.
                        // Re-check current tile (c,r) for potential vertical merge downwards.
                        if (r > 0 && board[c][r-1] == board[c][r]) {
                             r--; // Setup to re-evaluate (c,r-1) which is now the merged tile
                        }
                    }
                }
            }
        }
        if (resolve_loop_count >= max_resolve_loops) {
            // This cout is a warning for a potential infinite loop, might be okay to keep or log to a file if rare.
            // For now, let's comment it out for maximum speed during RL.
            // std::cout << "[MERGE_DEBUG] Warning: Exceeded max_resolve_loops. Breaking." << std::endl;
        }
    }
    return score_from_merges;
}

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
