#ifndef UTILS_HPP
#define UTILS_HPP

#include "game_defs.hpp" // For Tile
#include <map>           // For std::map (or other structure for the bag)
#include <random>        // For tile generation

// Generates a 'bag' of possible next tiles and their probabilities.
// The specifics of this function depend on your game's tile generation rules.
std::map<Tile, double> makeBag(Tile max_tile_on_board, std::mt19937& rng_engine);

// Helper to pick a tile from the bag based on probabilities
Tile pickTileFromBag(const std::map<Tile, double>& bag, std::mt19937& rng_engine);


#endif // UTILS_HPP
