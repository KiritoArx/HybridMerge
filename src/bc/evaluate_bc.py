import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO

# --- Add project root to sys.path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End of sys.path modification ---

from src.env.merge_env import MergeEnv # Your custom environment

# --- Configuration ---
PPO_MODEL_LOAD_PATH = os.path.join("models", "ppo_preview_v2_deep_explore_final.zip") # Make sure this points to the model you want to test
NUM_EVALUATION_EPISODES = 100 # Evaluate for 100 episodes for a good average
RENDER_DELAY_SECONDS = 0.0   # Adjust for watchability, 0 for max speed

def evaluate_ppo_agent(model_path, num_episodes=NUM_EVALUATION_EPISODES, render_delay=RENDER_DELAY_SECONDS):
    print(f"Evaluating PPO model.")
    print(f"Attempting to load model from: {model_path}")

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Please ensure it exists.")
        return

    env = MergeEnv(render_mode='human', curriculum_stage=None)
    print(f"Environment created. Action space: {env.action_space}, Obs space: {env.observation_space.shape}")

    try:
        model = PPO.load(model_path, env=env) 
        print(f"Successfully loaded PPO model from {model_path}")
    except Exception as e:
        print(f"ERROR: Could not load PPO model: {e}")
        env.close()
        return
    
    total_intrinsic_scores = []
    total_steps_taken = []

    for episode in range(num_episodes):
        print(f"\n--- Starting PPO Evaluation Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_intrinsic_score = 0
        episode_steps = 0

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            
            # The info from reset() or previous step might not have the score key we need for *this* step's log
            # We care about the score *after* the step is taken.
            # print(f"Step {episode_steps + 1} | Current Tile (from info): {info.get('current_tile_for_next_step')} | PPO Agent chose action: {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Correct key to get intrinsic score from the info dict returned by env.step()
            intrinsic_score_this_step = info.get("intrinsic_score_this_step", 0) 
            print(f"Step {episode_steps + 1} | Played Tile (was _current_game_tile_for_step): {env._current_game_tile_for_step if episode_steps ==0 else 'see_prev_next_tile'} | Action: {action} | Intrinsic Score this step: {intrinsic_score_this_step}")


            episode_intrinsic_score += intrinsic_score_this_step
            episode_steps += 1
            
            if render_delay > 0:
                time.sleep(render_delay)

            if terminated or truncated:
                print(f"Episode {episode + 1} finished after {episode_steps} steps.")
                print(f"Total Intrinsic Score for Episode {episode + 1}: {episode_intrinsic_score}")
                total_intrinsic_scores.append(episode_intrinsic_score)
                total_steps_taken.append(episode_steps)
                break
        
    env.close()

    if total_intrinsic_scores:
        print("\n--- PPO Agent Evaluation Summary ---")
        print(f"Number of episodes played: {len(total_intrinsic_scores)}")
        print(f"Average intrinsic score: {np.mean(total_intrinsic_scores):.2f}")
        print(f"Min intrinsic score: {np.min(total_intrinsic_scores):.2f}")
        print(f"Max intrinsic score: {np.max(total_intrinsic_scores):.2f}")
        print(f"Average steps per episode: {np.mean(total_steps_taken):.2f}")
    else:
        print("No episodes were completed for evaluation.")

if __name__ == "__main__":
    evaluate_ppo_agent(PPO_MODEL_LOAD_PATH)
