#include "merge.hpp"     // For Board, dropAndResolve, gameOver, maxTile, columnFull, printBoard
#include "ai.hpp"        // For findBestMove (your new AI)
#include "utils.hpp"     // For pickTileFromBag, makeBag
#include "game_defs.hpp" // For COLUMN_COUNT, MAX_HEIGHT, Tile

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>       // For std::random_device, std::mt19937
#include <sstream>      // For std::ostringstream

// Function to serialize the board to a flat JSON array string
// e.g., "[t1_col0, t2_col0, ..., t1_col1, t2_col1, ...]"
// Tiles are 0-padded up to MAX_HEIGHT for consistent vector length.
std::string serializeBoardFlat(const Board& board) {
    std::ostringstream oss;
    oss << "[";
    bool first_overall = true;
    for (int c = 0; c < COLUMN_COUNT; ++c) {
        for (int r = 0; r < MAX_HEIGHT; ++r) {
            if (!first_overall) {
                oss << ",";
            }
            // Ensure column c exists before trying to access its elements
            if (c < static_cast<int>(board.size()) && r < static_cast<int>(board[c].size())) {
                oss << board[c][r];
            } else {
                oss << 0; // Padding for empty cells or non-existent elements
            }
            first_overall = false;
        }
    }
    oss << "]";
    return oss.str();
}


int main() {
    // std::cout << "[DEBUG] expert_logger (using New 2-Ply AI) started." << std::endl;

    std::random_device rd;
    std::mt19937 rng_engine(rd()); // RNG for tile generation

    const int NUM_EPISODES = 10000; // Increased to generate a larger dataset
    const int AI_DEPTH = 2;      // Depth for the AI (your new AI uses a fixed 2-ply internally)

    std::string output_file_path = "data/expert_games.jsonl"; 
    // std::cout << "[DEBUG] Attempting to open " << output_file_path << std::endl;

    std::ofstream outfile(output_file_path);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open " << output_file_path << " for writing." << std::endl;
        std::cerr << "Please ensure the 'data' directory exists relative to the executable's CWD." << std::endl;
        return 1;
    }
    // std::cout << "[DEBUG] Successfully opened " << output_file_path << " for writing." << std::endl;

    long total_steps_logged = 0;
    long total_score_all_episodes = 0;

    for (int i = 0; i < NUM_EPISODES; ++i) {
        // std::cout << "[DEBUG] Starting Episode " << i + 1 << "/" << NUM_EPISODES << std::endl;
        Board board(COLUMN_COUNT); // Initialize empty board (vector of empty columns)
        int episode_score = 0;
        int steps_this_episode = 0;

        Tile currentTileForExpert = pickTileFromBag(makeBag(0, rng_engine), rng_engine); 
        Tile previewTileForExpert = pickTileFromBag(makeBag(0, rng_engine), rng_engine); 

        while (!gameOver(board)) {
            // std::cout << "[DEBUG] Episode " << i + 1 << ", Step " << steps_this_episode + 1 << std::endl;
            // printBoard(board); // For C++ side board visualization during data generation

            std::string state_before_action_str = serializeBoardFlat(board);
            
            Tile expert_sees_current = currentTileForExpert;
            Tile expert_sees_preview = previewTileForExpert;

            int expert_action = findBestMove(board, expert_sees_current, expert_sees_preview, AI_DEPTH);
            
            if (expert_action < 0 || expert_action >= COLUMN_COUNT || columnFull(board, expert_action)) {
                 std::cout << "[WARNING] AI suggested invalid or full column: " << expert_action 
                           << ". Attempting fallback to first available column." << std::endl;
                 bool found_fallback = false;
                 for(int fallback_col = 0; fallback_col < COLUMN_COUNT; ++fallback_col) {
                     if(!columnFull(board, fallback_col)) {
                         expert_action = fallback_col;
                         found_fallback = true;
                         break;
                     }
                 }
                 if (!found_fallback) {
                     std::cout << "[ERROR] No valid column to place tile. Ending episode prematurely." << std::endl;
                     break; 
                 }
            }

            int reward_from_merges = dropAndResolve(board, expert_sees_current, expert_action);
            episode_score += reward_from_merges; 

            std::string next_state_str = serializeBoardFlat(board);
            bool done = gameOver(board);
            
            outfile << "{";
            outfile << "\"state\":" << state_before_action_str << ",";
            outfile << "\"current_tile_for_decision\":" << expert_sees_current << ","; 
            outfile << "\"preview_tile_for_decision\":" << expert_sees_preview << ","; 
            outfile << "\"action\":" << expert_action << ",";
            outfile << "\"reward\":" << reward_from_merges << ","; 
            outfile << "\"next_state\":" << next_state_str << ",";
            outfile << "\"done\":" << (done ? "true" : "false");
            outfile << "}\n";
            outfile.flush(); 
            total_steps_logged++;

            currentTileForExpert = previewTileForExpert; 
            previewTileForExpert = pickTileFromBag(makeBag(maxTile(board), rng_engine), rng_engine); 

            steps_this_episode++;
            if (done) {
                break;
            }
        }
        total_score_all_episodes += episode_score;
        std::cout << "Episode " << i + 1 << "/" << NUM_EPISODES << " finished. Score: " << episode_score 
                  << ", Steps in episode: " << steps_this_episode << std::endl; 
    }

    outfile.close();
    std::cout << "\nExpert data generation complete." << std::endl;
    std::cout << "Total episodes run: " << NUM_EPISODES << std::endl;
    std::cout << "Total steps logged: " << total_steps_logged << std::endl;
    if (NUM_EPISODES > 0) { 
        double avg_score = static_cast<double>(total_score_all_episodes) / NUM_EPISODES;
        std::cout << "Average score per episode: " << avg_score << std::endl;
    }
    std::cout << "Data saved to " << output_file_path << std::endl;
    return 0;
}
