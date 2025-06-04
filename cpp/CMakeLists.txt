cmake_minimum_required(VERSION 3.12) 
project(HybridMergeAI_CPP)

set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Pybind11 ---
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# --- Source Files ---
set(CORE_AI_SOURCES
    ai.cpp # <<< ADDED ai.cpp here or ensure it's part of GAME_ENGINE_SOURCES
)

set(GAME_ENGINE_SOURCES
    merge.cpp
    utils.cpp
    game.cpp 
    ${CORE_AI_SOURCES} # <<< INCLUDE AI SOURCES FOR THE PYTHON MODULE
)

# --- Executable for expert_logger ---
# expert_logger also needs ai.cpp, merge.cpp, utils.cpp
add_executable(expert_logger
    expert_logger.cpp
    # GAME_ENGINE_SOURCES already includes ai.cpp, merge.cpp, utils.cpp, game.cpp
    # but expert_logger doesn't need game.cpp or bindings.cpp.
    # Let's list its specific dependencies for clarity:
    merge.cpp
    utils.cpp
    ai.cpp 
)

# --- Python Module ---
# The name "_merge_engine" here MUST match the first argument to PYBIND11_MODULE in bindings.cpp
# It needs bindings.cpp and all the game logic including ai.cpp (because game.cpp calls findBestMove)
pybind11_add_module(_merge_engine 
    bindings.cpp 
    game.cpp     # Game class implementation
    merge.cpp    # Core merge logic used by Game
    utils.cpp    # Utilities used by Game (e.g., for tile generation)
    ai.cpp       # AI logic (findBestMove) called by Game
)

# Optional: Set build type for optimization if not set by user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release) # Or Debug for development
endif()

# Enable warnings (good practice)
if(MSVC)
  add_compile_options(/W4 /EHsc) 
else()
  add_compile_options(-Wall -Wextra -pedantic -fPIC) 
endif()

