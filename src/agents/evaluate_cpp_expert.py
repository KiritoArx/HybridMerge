import os
import sys
import time
import numpy as np

# --- Add project root to sys.path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End of sys.path modification ---

from src.env.merge_env import MergeEnv # Your custom environment

# --- Configuration ---
NUM_EVALUATION_EPISODES = 100 # Evaluate for more episodes
RENDER_DELAY_SECONDS = 0.05   # Faster rendering for C++ expert, or 0 for no delay
AI_DEPTH_FOR_EXPERT = 2       # Depth to pass to C++ expert's findBestMove

def evaluate_cpp_expert(num_episodes=NUM_EVALUATION_EPISODES, render_delay=RENDER_DELAY_SECONDS):
    print(f"Evaluating C++ Expert AI ('New 2-Ply AI' via game_engine).")

    # Initialize the environment
    env = MergeEnv(render_mode='human', shaping_strength=0) # Shaping strength doesn't affect C++ expert's score here
    print(f"Environment created. Action space: {env.action_space}, Obs space: {env.observation_space.shape}")
    
    total_intrinsic_scores = []
    total_steps_taken = []

    for episode in range(num_episodes):
        print(f"\n--- Starting C++ Expert Evaluation Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset() # This sets env._current_game_tile via C++ Game's current_tile_
        
        terminated = False
        truncated = False
        episode_intrinsic_score = 0
        episode_steps = 0

        while not (terminated or truncated):
            # Get the tile the C++ expert should consider (which is also what player would play)
            tile_to_play_for_expert_and_move = env._current_game_tile # Accessing MergeEnv's current tile directly
                                                                      # as C++ Game's get_action_from_cpp_expert uses its own internal tiles

            # Get action from the C++ expert via the game_engine
            # The C++ Game's get_action_from_cpp_expert will use its own current_tile_ and preview_tile_
            # which were set during its last advance or reset.
            action = env.game_engine.get_action_from_cpp_expert(AI_DEPTH_FOR_EXPERT)
            
            # The tile used by the expert internally for decision was its own current_tile_.
            # The tile we pass to env.step should be what MergeEnv considers the current tile for the player.
            # This is already managed by env._current_game_tile after reset/step.
            
            print(f"Step {episode_steps + 1} | C++ Expert considers its current_tile={env.game_engine.get_current_tile_for_player()} "
                  f" (and its internal preview) | Chose action: {action}")
            print(f"   (MergeEnv's _current_game_tile to be played: {env._current_game_tile})")


            # Now, env.step will use env._current_game_tile to call game_engine.make_move
            obs, reward, terminated, truncated, info = env.step(action)
            # Rendering is handled within env.step()
            
            episode_intrinsic_score += info.get("intrinsic_score_step", 0) 
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
        print("\n--- C++ Expert AI Evaluation Summary ---")
        print(f"Number of episodes played: {len(total_intrinsic_scores)}")
        print(f"Average intrinsic score: {np.mean(total_intrinsic_scores):.2f}")
        print(f"Min intrinsic score: {np.min(total_intrinsic_scores):.2f}")
        print(f"Max intrinsic score: {np.max(total_intrinsic_scores):.2f}")
        print(f"Average steps per episode: {np.mean(total_steps_taken):.2f}")
    else:
        print("No C++ expert episodes were completed for evaluation.")

if __name__ == "__main__":
    evaluate_cpp_expert()
