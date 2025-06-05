import gymnasium as gym
from gymnasium import spaces
import numpy as np
# import _merge_engine # Moved import after sys.path debug
import json # For curriculum loading
import os   # For curriculum loading
import math # For math.log2
import sys  # For sys.path modification

# --- Add this module's directory to sys.path to find _merge_engine.pyd ---
MERGE_ENV_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if MERGE_ENV_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, MERGE_ENV_SCRIPT_DIR)
# --- End of sys.path modification for _merge_engine ---

import _merge_engine # Your compiled C++ module

class MergeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None, curriculum_stage=None, curriculum_file="configs/curriculum.json", shaping_strength=1.0): 
        super().__init__()
        
        self.game_engine = _merge_engine.Game()
        self.column_count = _merge_engine.COLUMN_COUNT
        self.max_height = _merge_engine.MAX_HEIGHT
        self.board_size = self.column_count * self.max_height

        self.observation_shape = (self.board_size + 2,) 
        self.max_tile_val_in_obs = 16384 
        self.observation_space = spaces.Box(low=0, high=self.max_tile_val_in_obs,
                                            shape=self.observation_shape,
                                            dtype=np.float32) 
        
        self.action_space = spaces.Discrete(self.column_count)
        
        self.render_mode = render_mode
        self._current_game_tile_for_step = None 
        self._preview_game_tile_for_obs = None  
        self._current_score_from_merges = 0 
        self._previous_max_tile = 0 
        self.shaping_strength = shaping_strength

        self.base_reward_weights = { # Using Re-balanced Rewards
            "intrinsic_score": 1.0, "empty_cells": 0.015,          
            "max_tile_value_log": 0.05, "max_tile_increase": 1.5,      
            "height_penalty_factor": -0.15, "future_penalty_factor": -0.3,   
            "roughness_penalty": -0.02, "adjacent_pairs_bonus": 0.03    
        }
        self.game_over_penalty = -10.0 

        self.curriculum_config = None
        if curriculum_stage: self._load_curriculum(curriculum_file, curriculum_stage)
        # print(f"MergeEnv initialized. Obs dtype: {self.observation_space.dtype}. Shaping strength: {self.shaping_strength}.")

    # --- New methods for MCTS state synchronization ---
    def get_serializable_game_state(self):
        """Returns a dictionary of the core game state needed to reconstruct it."""
        return {
            "board_flat": self.game_engine.get_flat_state(), # Already a list of ints
            "current_tile": self._current_game_tile_for_step, # Current tile for Python side
            "preview_tile": self._preview_game_tile_for_obs,   # Preview tile for Python side
            "previous_max_tile_for_reward": self._previous_max_tile # Needed for reward calc continuity
        }

    def set_serializable_game_state(self, state_data):
        """Sets the game state from a dictionary obtained via get_serializable_game_state."""
        self.game_engine.set_full_game_state_from_flat(
            state_data["board_flat"],
            state_data["current_tile"],
            state_data["preview_tile"]
        )
        # Update Python-side tracking variables
        self._current_game_tile_for_step = state_data["current_tile"]
        self._preview_game_tile_for_obs = state_data["preview_tile"]
        self._previous_max_tile = state_data["previous_max_tile_for_reward"]
        self._current_score_from_merges = 0 # Reset score for the new state
        # print(f"[MergeEnv DEBUG] State set. Current tile for step: {self._current_game_tile_for_step}, Preview for obs: {self._preview_game_tile_for_obs}")
    # --- End of new MCTS methods ---

    def _load_curriculum(self, file_path, stage_name): # (No change from before)
        try:
            script_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
            abs_path = os.path.join(project_root, file_path)
            with open(abs_path, 'r') as f: all_curriculum = json.load(f)
            self.curriculum_config = all_curriculum.get(stage_name)
            if self.curriculum_config: print(f"Loaded curriculum: {stage_name}")
            else: print(f"Warning: Curriculum stage '{stage_name}' not found.")
        except Exception as e: print(f"Warning: Error loading curriculum: {e}")

    def _calculate_reward_components(self, flat_state_np, previous_max_tile_val): # (No change from before)
        components = {}
        current_max_tile_val = np.max(flat_state_np) if flat_state_np.size > 0 else 0 
        components["empty_cells"] = np.sum(flat_state_np == 0)
        components["max_tile_value_log_component"] = math.log2(current_max_tile_val + 1) if current_max_tile_val > 0 else 0
        increase_in_max_tile = max(0, current_max_tile_val - previous_max_tile_val)
        components["max_tile_increase_component"] = increase_in_max_tile
        heights = np.zeros(self.column_count, dtype=int)
        if flat_state_np.size > 0: 
            board_reshaped_for_heights = flat_state_np.reshape((self.column_count, self.max_height))
            heights = np.count_nonzero(board_reshaped_for_heights, axis=1)
        danger_sum = np.sum(np.maximum(0, heights - (self.max_height - 2)))
        future_sum = np.sum(heights == (self.max_height - 1))
        components["height_penalty_value"] = danger_sum 
        components["future_penalty_value"] = future_sum 
        roughness = np.sum(np.abs(np.diff(heights))) if self.column_count > 1 else 0
        components["roughness"] = roughness
        pairs = 0
        if flat_state_np.size > 0:
            board_2d_visual = np.zeros((self.max_height, self.column_count), dtype=int)
            for c_idx in range(self.column_count):
                col_data_bottom_up = flat_state_np[c_idx * self.max_height : (c_idx + 1) * self.max_height]
                actual_tiles_in_col = 0
                for r_data_idx, tile_val in enumerate(col_data_bottom_up):
                    if tile_val > 0:
                        visual_row_idx_from_bottom = actual_tiles_in_col 
                        board_2d_visual[self.max_height - 1 - visual_row_idx_from_bottom, c_idx] = tile_val
                        actual_tiles_in_col += 1
            horizontal_matches = (board_2d_visual[:, :-1] == board_2d_visual[:, 1:]) & (board_2d_visual[:, :-1] != 0)
            pairs += np.sum(horizontal_matches)
            vertical_matches = (board_2d_visual[:-1, :] == board_2d_visual[1:, :]) & (board_2d_visual[:-1, :] != 0)
            pairs += np.sum(vertical_matches)
        components["adjacent_pairs"] = pairs
        return components, current_max_tile_val

    def _get_obs(self): 
        flat_board = np.array(self.game_engine.get_flat_state(), dtype=np.int32)
        obs = np.concatenate((flat_board, 
                              np.array([self._current_game_tile_for_step if self._current_game_tile_for_step is not None else 0], dtype=np.int32),
                              np.array([self._preview_game_tile_for_obs if self._preview_game_tile_for_obs is not None else 0], dtype=np.int32)
                             ))
        return obs.astype(np.float32) 

    def _get_info(self): 
        info_dict = {"current_tile_for_next_step": self._current_game_tile_for_step, 
                     "next_preview_tile": self._preview_game_tile_for_obs, 
                     "intrinsic_score_this_step": self._current_score_from_merges}
        return info_dict

    def reset(self, seed=None, options=None): 
        super().reset(seed=seed)
        self.game_engine.reset() 
        self._current_game_tile_for_step = self.game_engine.get_current_tile_for_player()
        self._preview_game_tile_for_obs = self.game_engine.get_preview_tile_for_player()
        self._current_score_from_merges = 0
        initial_board_obs_part = np.array(self.game_engine.get_flat_state(), dtype=np.int32)
        self._previous_max_tile = np.max(initial_board_obs_part) if initial_board_obs_part.size > 0 else 0
        observation = self._get_obs() 
        info = self._get_info() 
        if self.render_mode == "human": self.render()
        return observation, info

    def step(self, action):
        if not isinstance(action, (int, np.integer)):
            action = int(action.item())

        tile_played_this_step = self._current_game_tile_for_step 
        self._current_score_from_merges = self.game_engine.make_move(action, tile_played_this_step) 
        terminated = self.game_engine.is_game_over()
        current_board_flat_state_np = np.array(self.game_engine.get_flat_state(), dtype=np.int32)
        
        reward_components, current_max_val = self._calculate_reward_components(current_board_flat_state_np, self._previous_max_tile)
        
        shaped_reward = self._current_score_from_merges * self.base_reward_weights["intrinsic_score"] 
        for component_name, component_value in reward_components.items():
            if component_name in self.base_reward_weights and self.base_reward_weights[component_name] != 0:
                 if component_name != "intrinsic_score": 
                    shaped_reward += component_value * self.base_reward_weights[component_name] * self.shaping_strength
        
        if terminated:
            shaped_reward += self.game_over_penalty 

        self._previous_max_tile = current_max_val 

        if not terminated:
            self.game_engine.advance_tile_queue_for_player() 
            self._current_game_tile_for_step = self.game_engine.get_current_tile_for_player() 
            self._preview_game_tile_for_obs = self.game_engine.get_preview_tile_for_player()   
        else:
            self._current_game_tile_for_step = -1 
            self._preview_game_tile_for_obs = -1
            
        observation = self._get_obs() 
        info = {"current_tile_for_next_step": self._current_game_tile_for_step, 
                "next_preview_tile": self._preview_game_tile_for_obs,
                "intrinsic_score_this_step": self._current_score_from_merges,
                "shaped_reward_components": reward_components, 
                "shaping_strength": self.shaping_strength, 
                "final_step_reward": shaped_reward}
        truncated = False
        if self.render_mode == "human": self.render()
        return observation, shaped_reward, terminated, truncated, info

    def render(self): 
        if self.render_mode == 'ansi' or self.render_mode == 'human':
            flat_board_for_render = np.array(self.game_engine.get_flat_state(), dtype=np.int32)
            tile_to_play_for_render = self._current_game_tile_for_step
            preview_for_render = self._preview_game_tile_for_obs
            board_str = f"Current Tile for Agent to Play: {tile_to_play_for_render}\n"
            board_str += f"Preview Tile for Agent (in next obs): {preview_for_render}\n"
            board_str += "Board (top-down view):\n"
            for r_visual_top_down in range(self.max_height):
                row_str_parts = []
                for c_visual in range(self.column_count):
                    r_data_bottom_up = (self.max_height - 1) - r_visual_top_down
                    idx_in_flat_state = c_visual * self.max_height + r_data_bottom_up
                    tile_value = flat_board_for_render[idx_in_flat_state] if idx_in_flat_state < len(flat_board_for_render) else 0
                    cell_str = f"{tile_value:^6}" if tile_value != 0 else f"{'.':^6}"
                    row_str_parts.append(cell_str)
                board_str += " ".join(row_str_parts) + "\n"
            board_str += "-" * (self.column_count * 6 + (self.column_count - 1)) + "\n" 
            col_labels = [f"{'C'+str(c_label):^6}" for c_label in range(self.column_count)]
            board_str += " ".join(col_labels) + "\n"
            if self.render_mode == 'human': print(board_str)
            elif self.render_mode == 'ansi': return board_str 
        elif self.render_mode == 'rgb_array':
            return np.zeros((100, 100, 3), dtype=np.uint8) 

    def close(self):
        print("MergeEnv closed.")

if __name__ == '__main__': 
    print("Creating MergeEnv instance (with get/set state for MCTS)...") 
    env = MergeEnv(render_mode='human', shaping_strength=0.75) 
    print(f"Initial shaping strength: {env.shaping_strength}")
    print(f"Observation space dtype: {env.observation_space.dtype}") 
    
    # Test get_serializable_game_state and set_serializable_game_state
    print("\n--- Testing State Serialization ---")
    obs1, info1 = env.reset()
    state_data1 = env.get_serializable_game_state()
    print(f"State data 1: {state_data1}")

    # Make a few random moves
    for _ in range(3):
        action = env.action_space.sample()
        env.step(action)
    state_data2_before_set = env.get_serializable_game_state()
    print(f"State data 2 (after some moves): {state_data2_before_set}")
    
    # Set state back to state_data1
    env.set_serializable_game_state(state_data1)
    state_data3_after_set = env.get_serializable_game_state()
    print(f"State data 3 (after setting back to state 1): {state_data3_after_set}")
    
    # Verify by comparing board parts (tiles might be different due to rng in C++ reset if not seeded)
    # but board should match
    if np.array_equal(state_data1["board_flat"], state_data3_after_set["board_flat"]):
        print("Board part of state set/get successful!")
    else:
        print("Error: Board part of state set/get mismatch!")
    if state_data1["current_tile"] == state_data3_after_set["current_tile"] and \
       state_data1["preview_tile"] == state_data3_after_set["preview_tile"]:
        print("Tile parts of state set/get successful!")
    else:
        print("Error: Tile parts of state set/get mismatch! This might be okay if C++ tile generation in set_state is independent.")


    print("\nChecking environment with stable_baselines3.common.env_checker...")
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env) 
        print("SUCCESS: stable_baselines3.common.env_checker passed!")
    except ImportError: print("stable_baselines3.common.env_checker not available.")
    except Exception as e: print(f"ERROR: stable_baselines3.common.env_checker failed: {e}")
    env.close()
    print("\n--- Full game test run finished ---")
