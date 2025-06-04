#include "game.hpp"
#include <iostream> 

Game::Game() : rng_engine_(std::random_device{}()) {
    reset();
}

void Game::reset() {
    board_ = Board(COLUMN_COUNT, Column()); 
    for(int c=0; c < COLUMN_COUNT; ++c) {
        board_[c].clear(); 
    }
    // Initialize both current and preview tiles for the player sequence
    Tile board_max_tile_on_empty_board = 0; // For initial empty board
    current_tile_ = pickTileFromBag(makeBag(board_max_tile_on_empty_board, rng_engine_), rng_engine_);
    preview_tile_ = pickTileFromBag(makeBag(board_max_tile_on_empty_board, rng_engine_), rng_engine_);
    // std::cout << "[Game C++] Board reset. Player Current: " << current_tile_ << ", Player Preview: " << preview_tile_ << std::endl;
}

int Game::make_move(int column, Tile tile_value_to_play) {
    if (column < 0 || column >= COLUMN_COUNT || columnFull(board_, column)) {
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
            if (r < static_cast<int>(board_[c].size())) {
                flat_state.push_back(board_[c][r]);
            } else {
                flat_state.push_back(0); 
            }
        }
    }
    return flat_state;
}

bool Game::is_game_over() const {
    return gameOver(board_);
}

Tile Game::get_current_tile_for_player() const {
    return current_tile_;
}

Tile Game::get_preview_tile_for_player() const { // <<< NEW METHOD IMPLEMENTATION
    return preview_tile_;
}

void Game::advance_tile_queue_for_player() {
    current_tile_ = preview_tile_; // Old preview becomes current for player
    // Generate a new preview tile based on current board state
    preview_tile_ = pickTileFromBag(makeBag(maxTile(board_), rng_engine_), rng_engine_);
}

int Game::get_action_from_cpp_expert(int ai_depth) {
    // For the C++ expert, it might need its own view of current/preview if its
    // interaction model is different. Here, we assume it uses the game's main current/preview.
    // If expert_logger had a different tile sequence, this needs care.
    // For now, this uses the same current/preview as the player would see for their turn.
    return findBestMove(this->board_, this->current_tile_, this->preview_tile_, ai_depth);
}
