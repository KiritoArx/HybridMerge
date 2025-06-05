#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "game.hpp"       
#include "game_defs.hpp"  

namespace py = pybind11;

PYBIND11_MODULE(_merge_engine, m) { 
    m.doc() = "Pybind11 bindings for the C++ Tile-Merge game engine"; 

    m.attr("COLUMN_COUNT") = py::int_(COLUMN_COUNT);
    m.attr("MAX_HEIGHT") = py::int_(MAX_HEIGHT);

    py::class_<Game>(m, "Game")
        .def(py::init<>()) 
        .def("reset", &Game::reset, "Resets the game to an initial state.")
        .def("make_move", &Game::make_move, "Makes a move in the specified column with the given tile.",
             py::arg("column"), py::arg("tile_value"))
        .def("get_flat_state", &Game::get_flat_state, 
             "Returns the current board state as a flat list (size COLUMN_COUNT * MAX_HEIGHT).")
        .def("is_game_over", &Game::is_game_over, "Checks if the game is over.")
        .def("is_column_full", &Game::is_column_full, "Checks if a specific column is full.", py::arg("column"))
        .def("get_current_tile_for_player", &Game::get_current_tile_for_player, "Gets the current tile for the player/RL agent.")
        .def("get_preview_tile_for_player", &Game::Game::get_preview_tile_for_player, "Gets the upcoming preview tile for the player/RL agent.")
        .def("advance_tile_queue_for_player", &Game::advance_tile_queue_for_player, "Advances to the next tile for the player/RL agent.")
        .def("get_action_from_cpp_expert", &Game::get_action_from_cpp_expert, 
             "Gets an action suggestion from the internal C++ AI expert.",
             py::arg("ai_depth"))
        .def("set_full_game_state_from_flat", &Game::set_full_game_state_from_flat, // <<< ADDED
             "Sets the entire game state from a flat board list, current tile, and preview tile.",
             py::arg("flat_board_data"), py::arg("new_current_tile"), py::arg("new_preview_tile")); 
}
