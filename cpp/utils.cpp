#include "utils.hpp"
#include <vector>       // For std::discrete_distribution
#include <numeric>      // For std::accumulate in pickTileFromBag (optional)

// Generates a 'bag' of possible next tiles and their probabilities.
std::map<Tile, double> makeBag(Tile max_tile_on_board, std::mt19937& rng_engine) {
    std::map<Tile, double> bag;
    // Example tile generation logic:
    // Adjust this to match your game's specific rules.
    // This example spawns 2s and 4s, and occasionally 8s if higher tiles are present.
    
    bag[2] = 0.70; // 70% chance for tile 2
    bag[4] = 0.25; // 25% chance for tile 4

    if (max_tile_on_board >= 16) { // If a 16 or higher is on board
        bag[8] = 0.05; // 5% chance for tile 8
    } else if (max_tile_on_board >= 8) {
        // If max is 8 or more, but less than 16, maybe adjust probabilities
        // For simplicity, we'll keep it as above, or you can add more rules.
        // To ensure probabilities sum to 1 if 8 is not added:
        if (bag.find(8) == bag.end()) {
             // Redistribute the 0.05 if 8 is not added, or simply normalize later.
             // For now, we assume the AI's expectimax handles non-normalized probabilities if sum is consistent.
        }
    }
    // It's good practice to ensure probabilities sum to 1.0 if your expectimax relies on it.
    // The provided ai.cpp expecti function normalizes by prob_sum, so it's robust.
    return bag;
}

// Helper to pick a tile from the bag based on probabilities
Tile pickTileFromBag(const std::map<Tile, double>& bag, std::mt19937& rng_engine) {
    if (bag.empty()) {
        return 2; // Default tile if bag is empty for some reason
    }

    std::vector<Tile> tiles;
    std::vector<double> probabilities;
    for (const auto& pair : bag) {
        tiles.push_back(pair.first);
        probabilities.push_back(pair.second);
    }

    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    return tiles[dist(rng_engine)];
}
