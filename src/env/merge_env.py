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

        self.action_space = spaces.Discrete(self.column_count)
        self.observation_shape = (self.max_height * self.column_count + 2,) # board + current + preview
        self.observation_space = spaces.Box(low=0, high=16384, 
                                            shape=self.observation_shape,
                                            dtype=np.int32)
        
        self.render_mode = render_mode
        self._current_game_tile_for_step = None 
        self._preview_game_tile_for_obs = None  
        self._current_score_from_merges = 0 
        self._previous_max_tile = 0 # Still useful for potential analysis, but not heavily rewarded
        self.shaping_strength = shaping_strength # Will scale the few remaining shaping terms

        # --- Phase 5: Simplified Base Reward Shaping Weights ---
        # Focus on intrinsic score, with minimal essential shaping.
        self.base_reward_weights = {
            "intrinsic_score": 1.0,        # Primary driver: actual game score from merges
            "empty_cells": 0.001,          # Very small bonus for open board
            "max_tile_value_log": 0.0,     # Disabled: Let agent learn value of high tiles via intrinsic score
            "max_tile_increase": 0.1,      # Small hint for making bigger tiles, but not dominant
            "height_penalty_factor": -0.05, # Reduced penalty for danger zone
            "future_penalty_factor": -0.1,  # Reduced penalty for nearly full columns
            "roughness_penalty": -0.00,     # Disabled: Let agent figure out board structure
            "adjacent_pairs_bonus": 0.00    # Disabled: Let agent learn merge setups
        }
        # Add a game over penalty directly in step if needed, or rely on PPO's value function for terminal states.
        self.game_over_penalty = -10.0 # A clear penalty for losing the game

        self.curriculum_config = None
        if curriculum_stage:
            self._load_curriculum(curriculum_file, curriculum_stage)

        print(f"MergeEnv initialized (Phase 5 - Simplified Rewards). Shaping strength: {self.shaping_strength}")

    def _load_curriculum(self, file_path, stage_name): # (No change)
        try:
            script_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
            abs_path = os.path.join(project_root, file_path)
            with open(abs_path, 'r') as f: all_curriculum = json.load(f)
            self.curriculum_config = all_curriculum.get(stage_name)
            if self.curriculum_config: print(f"Loaded curriculum: {stage_name}")
            else: print(f"Warning: Curriculum stage '{stage_name}' not found.")
        except Exception as e: print(f"Warning: Error loading curriculum: {e}")

    def _calculate_reward_components(self, flat_state_np, previous_max_tile_val): # (No change in calculation, only weights change)
        components = {}
        current_max_tile_val = np.max(flat_state_np) if flat_state_np.size > 0 else 0 
        components["empty_cells"] = np.sum(flat_state_np == 0)
        components["max_tile_value_log_component"] = math.log2(current_max_tile_val + 1) if current_max_tile_val > 0 else 0
        increase_in_max_tile = max(0, current_max_tile_val - previous_max_tile_val)
        components["max_tile_increase_component"] = increase_in_max_tile
        heights = np.zeros(self.column_count, dtype=int)
        if flat_state_np.size > 0: 
            for c in range(self.column_count):
                col_start_idx = c * self.max_height
                col_end_idx = (c + 1) * self.max_height
                column_tiles = flat_state_np[col_start_idx:col_end_idx]
                heights[c] = np.count_nonzero(column_tiles)
        danger_sum, future_sum, roughness = 0,0,0
        for h_idx, h_val in enumerate(heights):
            danger_sum += max(0, h_val - (self.max_height - 2)) 
            if h_val == self.max_height - 1: future_sum += 1
            if h_idx > 0: roughness += abs(h_val - heights[h_idx-1])
        components["height_penalty_value"] = danger_sum 
        components["future_penalty_value"] = future_sum 
        components["roughness"] = roughness
        board_2d = np.zeros((self.max_height, self.column_count), dtype=int)
        if flat_state_np.size > 0: 
            for c_idx in range(self.column_count):
                actual_tiles_in_col = 0
                for r_idx_in_col_data in range(self.max_height):
                    tile_val = flat_state_np[c_idx * self.max_height + r_idx_in_col_data]
                    if tile_val > 0:
                        visual_row = self.max_height - 1 - actual_tiles_in_col
                        board_2d[visual_row, c_idx] = tile_val
                        actual_tiles_in_col += 1
        pairs = 0
        for r_vis in range(self.max_height):
            for c_vis in range(self.column_count - 1):
                if board_2d[r_vis, c_vis] != 0 and board_2d[r_vis, c_vis] == board_2d[r_vis, c_vis+1]: pairs +=1
        for c_vis in range(self.column_count):
            for r_vis in range(self.max_height - 1):
                if board_2d[r_vis, c_vis] != 0 and board_2d[r_vis, c_vis] == board_2d[r_vis+1, c_vis]: pairs +=1
        components["adjacent_pairs"] = pairs
        return components, current_max_tile_val

    def _get_obs(self): # (No change)
        flat_board = np.array(self.game_engine.get_flat_state(), dtype=np.int32)
        obs = np.concatenate((flat_board, 
                              np.array([self._current_game_tile_for_step if self._current_game_tile_for_step is not None else 0], dtype=np.int32),
                              np.array([self._preview_game_tile_for_obs if self._preview_game_tile_for_obs is not None else 0], dtype=np.int32)
                             ))
        return obs

    def _get_info(self): # (No change)
        info_dict = {"current_tile_for_next_step": self._current_game_tile_for_step, 
                     "next_preview_tile": self._preview_game_tile_for_obs, 
                     "intrinsic_score_this_step": self._current_score_from_merges}
        return info_dict

    def reset(self, seed=None, options=None): # (No change)
        super().reset(seed=seed)
        self.game_engine.reset() 
        self._current_game_tile_for_step = self.game_engine.get_current_tile_for_player()
        self._preview_game_tile_for_obs = self.game_engine.get_preview_tile_for_player()
        self._current_score_from_merges = 0
        initial_board_obs = np.array(self.game_engine.get_flat_state(), dtype=np.int32)
        self._previous_max_tile = np.max(initial_board_obs) if initial_board_obs.size > 0 else 0
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
        current_flat_state_np_board_only = np.array(self.game_engine.get_flat_state(), dtype=np.int32)
        
        reward_components, current_max_val = self._calculate_reward_components(current_flat_state_np_board_only, self._previous_max_tile)
        
        shaped_reward = self._current_score_from_merges * self.base_reward_weights["intrinsic_score"] 
        # Apply scaled shaping for the few remaining terms
        for component_name, component_value in reward_components.items():
            if component_name in self.base_reward_weights and component_name != "intrinsic_score":
                shaped_reward += component_value * self.base_reward_weights[component_name] * self.shaping_strength
        
        if terminated:
            shaped_reward += self.game_over_penalty # Add a direct game over penalty

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
                "shaped_reward_components": reward_components, # For logging
                "shaping_strength": self.shaping_strength, 
                "final_step_reward": shaped_reward}
        truncated = False
        if self.render_mode == "human": self.render()
        return observation, shaped_reward, terminated, truncated, info

    def render(self): # (No change)
        if self.render_mode == 'ansi' or self.render_mode == 'human':
            flat_board_for_render = np.array(self.game_engine.get_flat_state(), dtype=np.int32)
            board_str = f"Current Tile for Agent to Play: {self._current_game_tile_for_step}\n"
            board_str += f"Preview Tile for Agent (in next obs): {self._preview_game_tile_for_obs}\n"
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
    print("Creating MergeEnv instance (Phase 5 - Simplified Rewards)...") 
    env = MergeEnv(render_mode='human', shaping_strength=0.1) # Test with low shaping
    print(f"Initial shaping strength: {env.shaping_strength}")
    
    print("\nChecking environment with stable_baselines3.common.env_checker...")
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env) 
        print("SUCCESS: stable_baselines3.common.env_checker passed!")
    except ImportError:
        print("stable_baselines3.common.env_checker not available.")
    except Exception as e:
        print(f"ERROR: stable_baselines3.common.env_checker failed: {e}")

    print("\n--- Starting a full game test run with Simplified Rewards ---") 
    obs, info = env.reset()
    terminated, truncated, episode_steps = False, False, 0
    max_steps_per_episode = 50 
    while not (terminated or truncated):
        action = env.action_space.sample() 
        print(f"\nStep {episode_steps + 1}: Action={action}, Playing tile={info.get('current_tile_for_next_step')}")
        obs, reward, terminated, truncated, info = env.step(action)
        episode_steps += 1
        if episode_steps >= max_steps_per_episode: truncated = True 
        if terminated or truncated: print(f"\n--- Episode Finished ---"); break 
    env.close()
    print("\n--- Full game test run finished ---")
