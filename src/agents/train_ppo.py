import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn # For LR schedule
import sys
from functools import partial 

# --- Add project root to sys.path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End of sys.path modification ---

from src.env.merge_env import MergeEnv 
from src.bc.train_bc import STATE_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, NUM_ACTIONS 

# --- Configuration ---
LOG_DIR = "ppo_tensorboard_logs/" 
MODEL_SAVE_DIR = "models/"

# --- Parameters for Phase 5 Training Run ---
# Start from the PPO agent that was trained with preview tiles and had a non-zero score.
# If ppo_preview_v2_deep_explore_final.zip (avg ~1607) is your best so far, use that.
# Or, if you have another model that learned with preview tiles and performed better, use that one.
# For this example, let's assume we're continuing from the one that achieved ~1607.
CONTINUE_TRAINING_FROM_MODEL = os.path.join(MODEL_SAVE_DIR, "ppo_preview_v2_deep_explore_final.zip") 
NEW_RUN_NAME_PREFIX = "ppo_phase5_simplified_reward_long" 
PPO_MODEL_SAVE_PATH_FINAL = os.path.join(MODEL_SAVE_DIR, f"{NEW_RUN_NAME_PREFIX}_final") 

ADDITIONAL_TIMESTEPS = 15_000_000  # Very long run for deep learning with simpler rewards
SHAPING_STRENGTH_FOR_ENV = 0.25    # Low strength for the few remaining shaping terms

# PPO Hyperparameters 
# Learning rate will use a linear schedule
INITIAL_LEARNING_RATE = 2e-4  # Starting learning rate
FINAL_LEARNING_RATE = 1e-5    # Learning rate will decay to this value
NUM_ENVIRONMENTS = 4         
N_STEPS = 2048               
BATCH_SIZE = 64                
N_EPOCHS = 10                  
GAMMA = 0.99                   
GAE_LAMBDA = 0.95              
CLIP_RANGE = 0.15              # Slightly smaller clip range for more stable fine-tuning
ENT_COEF = 0.0005              # Low entropy for exploitation, but not zero
VF_COEF = 0.5                  
MAX_GRAD_NORM = 0.5            

POLICY_KWARGS = dict(net_arch=dict(pi=[HIDDEN_SIZE_1, HIDDEN_SIZE_2], 
                                   vf=[HIDDEN_SIZE_1, HIDDEN_SIZE_2]))

def main():
    print(f"Starting PPO Phase 5 training run: {NEW_RUN_NAME_PREFIX}")
    print(f"Attempting to continue training from: {CONTINUE_TRAINING_FROM_MODEL}")
    print(f"Additional timesteps: {ADDITIONAL_TIMESTEPS}")
    print(f"Shaping strength for MergeEnv: {SHAPING_STRENGTH_FOR_ENV}")
    print(f"Entropy coefficient (ent_coef): {ENT_COEF}")
    print(f"Learning rate schedule: From {INITIAL_LEARNING_RATE} to {FINAL_LEARNING_RATE}")

    current_run_log_dir = os.path.join(LOG_DIR, NEW_RUN_NAME_PREFIX)
    os.makedirs(current_run_log_dir, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    env_kwargs = {'shaping_strength': SHAPING_STRENGTH_FOR_ENV, 'curriculum_stage': None}
    vec_env = make_vec_env(MergeEnv, n_envs=NUM_ENVIRONMENTS, env_kwargs=env_kwargs)
    print("Vectorized environments created with simplified reward shaping.")

    # Define learning rate schedule
    # Progress_remaining will go from 1.0 (start) to 0.0 (end of training)
    lr_schedule = get_linear_fn(start=INITIAL_LEARNING_RATE, end=FINAL_LEARNING_RATE, end_fraction=1.0)

    model_loaded_successfully = False
    if os.path.exists(CONTINUE_TRAINING_FROM_MODEL):
        print(f"Loading existing PPO model from {CONTINUE_TRAINING_FROM_MODEL}...")
        try:
            model = PPO.load(CONTINUE_TRAINING_FROM_MODEL, 
                             env=vec_env, 
                             tensorboard_log=current_run_log_dir,
                             # For continuing training, explicitly set new schedulable/tunable hyperparameters
                             learning_rate=lr_schedule, # Pass the schedule function
                             ent_coef=ENT_COEF,
                             clip_range=CLIP_RANGE, 
                             # Other core PPO parameters like n_steps, batch_size, n_epochs are often
                             # re-initialized by SB3 when you call .learn() or should be passed here if needed.
                             # Forcing them here ensures they are what we intend for this new phase.
                             n_steps=N_STEPS,
                             batch_size=BATCH_SIZE,
                             n_epochs=N_EPOCHS
                            )
            print(f"Model loaded. Parameters for this run: ent_coef={model.ent_coef}, clip_range={model.clip_range}")
            print(f"Learning rate will use a linear schedule from {INITIAL_LEARNING_RATE} to {FINAL_LEARNING_RATE}.")
            model_loaded_successfully = True
        except Exception as e:
            print(f"Error loading model from {CONTINUE_TRAINING_FROM_MODEL}: {e}")
            print("Cannot proceed. Exiting.")
            vec_env.close()
            return
    else:
        print(f"Base model not found at {CONTINUE_TRAINING_FROM_MODEL}. Cannot proceed. Exiting.")
        vec_env.close()
        return

    print(f"PPO agent ready for Phase 5 training. Using device: {model.device}")

    save_frequency = max(N_STEPS * NUM_ENVIRONMENTS * 100, 1_000_000) 
    checkpoint_callback = CheckpointCallback(save_freq=save_frequency,
                                             save_path=MODEL_SAVE_DIR,
                                             name_prefix=NEW_RUN_NAME_PREFIX + "_checkpoint")
    
    print(f"Model checkpoints will be saved every {save_frequency} total steps.")
    callbacks_list = [checkpoint_callback]

    print(f"Starting PPO model training (Phase 5) for {ADDITIONAL_TIMESTEPS} additional steps...")
    try:
        model.learn(total_timesteps=ADDITIONAL_TIMESTEPS, 
                    callback=callbacks_list, 
                    reset_num_timesteps=False) # Continue step count from loaded model
        
        print("PPO model training (Phase 5) finished.")
        model.save(PPO_MODEL_SAVE_PATH_FINAL)
        print(f"Final PPO model for run '{NEW_RUN_NAME_PREFIX}' saved to {PPO_MODEL_SAVE_PATH_FINAL}.zip")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        interrupted_save_path = os.path.join(MODEL_SAVE_DIR, f"{NEW_RUN_NAME_PREFIX}_interrupted")
        model.save(interrupted_save_path)
        print(f"Model progress saved to {interrupted_save_path}.zip")
    except Exception as e:
        print(f"An error occurred during PPO training: {e}")
        error_save_path = os.path.join(MODEL_SAVE_DIR, f"{NEW_RUN_NAME_PREFIX}_on_error")
        model.save(error_save_path)
        print(f"Model progress saved to {error_save_path}.zip")
    finally:
        vec_env.close()
        print("Vectorized environments closed.")

if __name__ == "__main__":
    main()
