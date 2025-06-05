#include "game.hpp"
#include <iostream> 
#include <stdexcept> // For std::runtime_error

Game::Game() : rng_engine_(std::random_device{}()) {
    reset();
}

void Game::reset() {
    board_ = Board(COLUMN_COUNT, Column()); 
    for(int c=0; c < COLUMN_COUNT; ++c) {
        board_[c].clear(); 
    }
    Tile board_max_tile_on_empty_board = 0; 
    current_tile_ = pickTileFromBag(makeBag(board_max_tile_on_empty_board, rng_engine_), rng_engine_);
    preview_tile_ = pickTileFromBag(makeBag(board_max_tile_on_empty_board, rng_engine_), rng_engine_);
}

int Game::make_move(int column, Tile tile_value_to_play) {
    if (column < 0 || column >= COLUMN_COUNT || ::columnFull(board_, column)) {
        return -1; 
    }
    int reward = dropAndResolve(board_, tile_value_to_play, column);
    return reward;
}

std::vector<Tile> Game::get_flat_state() const {
    std::vector<Tile> flat_state;
    flat_state.reserve(COLUMN_COUNT * MAX_HEIGHT);
    for (int c = 0; c < COLUMN_COUNT; ++c) {
        for (int r = 0; r < MAX_HEIGHT; ++r) {
            if (c < static_cast<int>(board_.size()) && r < static_cast<int>(board_[c].size())) {
                flat_state.push_back(board_[c][r]);
            } else {
                flat_state.push_back(0); 
            }
        }
    }
    return flat_state;
}

bool Game::is_game_over() const {
    return ::gameOver(board_);
}

bool Game::is_column_full(int col) const {
    if (col < 0 || col >= COLUMN_COUNT) return true;
    return ::columnFull(board_, col); 
}

Tile Game::get_current_tile_for_player() const {
    return current_tile_;
}

Tile Game::get_preview_tile_for_player() const { 
    return preview_tile_;
}

void Game::advance_tile_queue_for_player() {
    current_tile_ = preview_tile_; 
    preview_tile_ = pickTileFromBag(makeBag(maxTile(board_), rng_engine_), rng_engine_);
}

int Game::get_action_from_cpp_expert(int ai_depth) {
    return findBestMove(this->board_, this->current_tile_, this.preview_tile_, ai_depth);
}

// --- New method for setting state ---
void Game::set_full_game_state_from_flat(const std::vector<Tile>& flat_board_data, 
                                         Tile new_current_tile, 
                                         Tile new_preview_tile) {
    if (flat_board_data.size() != static_cast<size_t>(COLUMN_COUNT * MAX_HEIGHT)) {
        // Consider throwing an error or logging if dimensions don't match
        // For simplicity, this example assumes correct size.
        // You could add: throw std::runtime_error("Invalid flat_board_data size");
        std::cerr << "Error: Invalid flat_board_data size in set_full_game_state_from_flat." << std::endl;
        return;
    }

    board_.assign(COLUMN_COUNT, Column()); // Clear and resize board
    for (int c = 0; c < COLUMN_COUNT; ++c) {
        board_[c].clear(); // Ensure column is empty before populating
        for (int r_bottom_up = 0; r_bottom_up < MAX_HEIGHT; ++r_bottom_up) {
            Tile tile_val = flat_board_data[c * MAX_HEIGHT + r_bottom_up];
            if (tile_val != 0) {
                board_[c].push_back(tile_val);
            }
        }
    }

    current_tile_ = new_current_tile;
    preview_tile_ = new_preview_tile;
    // Note: This does not reset rng_engine_ or score. MCTS simulations typically
    // care about state transitions and terminal rewards, not intermediate C++ scores.
}
