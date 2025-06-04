#ifndef GAME_HPP
#define GAME_HPP

#include "game_defs.hpp" // For Board, Tile, COLUMN_COUNT, MAX_HEIGHT
#include "merge.hpp"     // For dropAndResolve, gameOver, etc.
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

    Tile get_current_tile_for_player() const; 
    Tile get_preview_tile_for_player() const;   // <<< NEW METHOD
    void advance_tile_queue_for_player();     

    int get_action_from_cpp_expert(int ai_depth);

private:
    Board board_;
    std::mt19937 rng_engine_;
    Tile current_tile_;
    Tile preview_tile_; 

    // void generate_new_tile_pair(); // This was for expert, reset/advance handles player tiles
};

#endif // GAME_HPP
