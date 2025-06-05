#ifndef GAME_HPP
#define GAME_HPP

#include "game_defs.hpp" // For Board, Tile, COLUMN_COUNT, MAX_HEIGHT
#include "merge.hpp"     // For dropAndResolve, gameOver, columnFull etc.
#include "utils.hpp"     // For pickTileFromBag, makeBag
#include "ai.hpp"        // To get findBestMove declaration
#include <vector>
#include <random>       // For std::mt19937

class Game {
public:
    Game(); // Constructor
    void reset();
    
    int make_move(int column, Tile current_tile_value); 
    std::vector<Tile> get_flat_state() const; 
    bool is_game_over() const;
    bool is_column_full(int col) const; 

    Tile get_current_tile_for_player() const; 
    Tile get_preview_tile_for_player() const;   
    void advance_tile_queue_for_player();     

    int get_action_from_cpp_expert(int ai_depth);

    // --- New methods for setting state ---
    // Takes a flat board representation
    void set_full_game_state_from_flat(const std::vector<Tile>& flat_board_data, 
                                       Tile new_current_tile, 
                                       Tile new_preview_tile);

private:
    Board board_;
    std::mt19937 rng_engine_; // rng_engine state is not typically part of what you'd set externally for MCTS sim
    Tile current_tile_;
    Tile preview_tile_; 
};

#endif // GAME_HPP
