import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import get_linear_fn 
import sys
from functools import partial 

# --- Add project root to sys.path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End of sys.path modification ---

from src.env.merge_env import MergeEnv 
from src.bc.train_bc import BCModel, STATE_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, NUM_ACTIONS 

# --- Configuration ---
LOG_DIR = "ppo_tensorboard_logs/" 
MODEL_SAVE_DIR = "models/"
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_models_phase5_scheduled_ent") 

# --- Parameters for Phase 5 Training Run ---
BC_MODEL_LOAD_PATH = os.path.join(MODEL_SAVE_DIR, "bc_policy_with_preview.pt") # Your ~96% BC model
NEW_RUN_NAME_PREFIX = "ppo_phase5_bc_warmstart_sched_ent_v2" # New name for this run
PPO_MODEL_SAVE_PATH_FINAL = os.path.join(MODEL_SAVE_DIR, f"{NEW_RUN_NAME_PREFIX}_final") 

TOTAL_TIMESTEPS = 15_000_000  # Very long run
SHAPING_STRENGTH_FOR_ENV = 0.75 # Start with moderate shaping, can be annealed later via callback if needed

# PPO Hyperparameters 
INITIAL_LEARNING_RATE = 2e-4  
FINAL_LEARNING_RATE = 2e-5    # Decay to a slightly lower final LR
INITIAL_ENT_COEF = 0.01       # Start with higher exploration
FINAL_ENT_COEF = 0.0001       # Decay to very low exploration
NUM_ENVIRONMENTS = 4         
N_STEPS = 4096               
BATCH_SIZE = 128               
N_EPOCHS = 10                  
GAMMA = 0.99                   
GAE_LAMBDA = 0.95              
CLIP_RANGE = 0.2 # Start with standard, can be scheduled too.             
VF_COEF = 0.5                  
MAX_GRAD_NORM = 0.5            
DEVICE_TO_USE = "cpu"

POLICY_KWARGS = dict(net_arch=dict(pi=[HIDDEN_SIZE_1, HIDDEN_SIZE_2], 
                                   vf=[HIDDEN_SIZE_1, HIDDEN_SIZE_2]))

def warm_start_ppo_from_bc(ppo_model, bc_model_path, bc_input_size, bc_hidden1, bc_hidden2, bc_output_size):
    # (Warm-start function remains the same as before)
    print(f"Attempting to warm-start PPO model from BC policy at: {bc_model_path}")
    print(f"Expected BC input size for PPO policy: {bc_input_size}")
    try:
        bc_policy = BCModel(bc_input_size, bc_hidden1, bc_hidden2, bc_output_size)
        bc_policy.load_state_dict(torch.load(bc_model_path, map_location=ppo_model.device))
        bc_policy.eval()
        print("BC policy (trained with preview tile) loaded successfully.")
        
        bc_sequential_network = bc_policy.network
        
        ppo_model.policy.mlp_extractor.policy_net[0].weight.data.copy_(bc_sequential_network[0].weight.data)
        ppo_model.policy.mlp_extractor.policy_net[0].bias.data.copy_(bc_sequential_network[0].bias.data)
        ppo_model.policy.mlp_extractor.policy_net[2].weight.data.copy_(bc_sequential_network[2].weight.data)
        ppo_model.policy.mlp_extractor.policy_net[2].bias.data.copy_(bc_sequential_network[2].bias.data)
        ppo_model.policy.action_net.weight.data.copy_(bc_sequential_network[4].weight.data)
        ppo_model.policy.action_net.bias.data.copy_(bc_sequential_network[4].bias.data)

        print("Successfully attempted to transfer weights from BC policy to PPO actor.")
    except FileNotFoundError:
        print(f"ERROR: BC model file not found at {bc_model_path}. PPO will train from scratch, which is not intended for this run configuration.")
        raise 
    except Exception as e:
        print(f"ERROR: Could not warm-start PPO from BC policy: {e}")
        print("PPO will train from scratch.")
        raise 

class EntropyScheduleCallback(BaseCallback):
    """
    A custom callback to linearly anneal the entropy coefficient (ent_coef).
    """
    def __init__(self, initial_ent_coef, final_ent_coef, total_training_timesteps, verbose=0):
        super(EntropyScheduleCallback, self).__init__(verbose)
        self.schedule_fn = get_linear_fn(start=initial_ent_coef, end=final_ent_coef, end_fraction=1.0)
        self.total_training_timesteps = total_training_timesteps

    def _on_step(self) -> bool:
        # Calculate current progress (0.0 to 1.0)
        progress = self.num_timesteps / self.total_training_timesteps
        # Ensure progress doesn't exceed 1.0 if training goes slightly beyond total_timesteps
        progress = min(progress, 1.0) 
        
        new_ent_coef = self.schedule_fn(1.0 - progress) # schedule_fn expects progress_remaining (1.0 -> 0.0)
        self.model.ent_coef = new_ent_coef
        
        # Log the current ent_coef to TensorBoard
        if self.num_timesteps % self.model.n_steps == 0: # Log every rollout collection
             self.logger.record("train/ent_coef_scheduled", new_ent_coef)
        return True

def main():
    print(f"Starting PPO Phase 5 training run: {NEW_RUN_NAME_PREFIX}")
    print(f"Warm-starting from BC model: {BC_MODEL_LOAD_PATH}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Shaping strength for MergeEnv: {SHAPING_STRENGTH_FOR_ENV}")
    print(f"Entropy coefficient schedule: From {INITIAL_ENT_COEF} to {FINAL_ENT_COEF}")
    print(f"Learning rate schedule: From {INITIAL_LEARNING_RATE} to {FINAL_LEARNING_RATE}")
    print(f"Device: {DEVICE_TO_USE}")

    current_run_log_dir = os.path.join(LOG_DIR, NEW_RUN_NAME_PREFIX)
    os.makedirs(current_run_log_dir, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_SAVE_PATH, exist_ok=True)

    env_kwargs = {'shaping_strength': SHAPING_STRENGTH_FOR_ENV, 'curriculum_stage': None}
    vec_env = make_vec_env(lambda: MergeEnv(**env_kwargs), 
                           n_envs=NUM_ENVIRONMENTS, 
                           vec_env_cls=SubprocVecEnv)
    print("Vectorized training environments created.")

    eval_env_kwargs = {'shaping_strength': SHAPING_STRENGTH_FOR_ENV, 'render_mode': None, 'curriculum_stage': None}
    eval_env = DummyVecEnv([lambda: Monitor(MergeEnv(**eval_env_kwargs))])
    print("Evaluation environment created.")

    lr_schedule = get_linear_fn(start=INITIAL_LEARNING_RATE, end=FINAL_LEARNING_RATE, end_fraction=1.0)
    
    print("Initializing a new PPO agent for warm-starting...")
    model = PPO("MlpPolicy", 
                vec_env, 
                verbose=1, 
                learning_rate=lr_schedule, 
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_RANGE, 
                ent_coef=INITIAL_ENT_COEF, # Start with initial, callback will anneal
                vf_coef=VF_COEF,
                max_grad_norm=MAX_GRAD_NORM,
                policy_kwargs=POLICY_KWARGS, 
                tensorboard_log=current_run_log_dir,
                device=DEVICE_TO_USE 
                )
    print(f"PPO agent initialized. Using device: {model.device}")

    if os.path.exists(BC_MODEL_LOAD_PATH):
        warm_start_ppo_from_bc(model, BC_MODEL_LOAD_PATH, STATE_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, NUM_ACTIONS)
    else:
        print(f"CRITICAL ERROR: BC model '{BC_MODEL_LOAD_PATH}' not found. Warm-start is essential for this run. Exiting.")
        vec_env.close(); eval_env.close(); return

    checkpoint_save_freq = max(N_STEPS * NUM_ENVIRONMENTS * 50, 1_000_000) 
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_save_freq,
                                             save_path=MODEL_SAVE_DIR,
                                             name_prefix=NEW_RUN_NAME_PREFIX + "_checkpoint")
    
    eval_freq = max(N_STEPS * NUM_ENVIRONMENTS * 10, 250000) 
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=BEST_MODEL_SAVE_PATH, 
                                 log_path=current_run_log_dir, 
                                 eval_freq=eval_freq,
                                 n_eval_episodes=20, 
                                 deterministic=True, 
                                 render=False)
    
    # Entropy schedule callback
    entropy_schedule_callback = EntropyScheduleCallback(INITIAL_ENT_COEF, FINAL_ENT_COEF, TOTAL_TIMESTEPS)
    
    print(f"Model checkpoints will be saved every {checkpoint_save_freq} total steps.")
    print(f"Model will be evaluated every {eval_freq} total steps, best model saved to {BEST_MODEL_SAVE_PATH}")
    callbacks_list = [checkpoint_callback, eval_callback, entropy_schedule_callback] # Added entropy scheduler

    print(f"Starting PPO model training (Phase 5) for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, 
                    callback=callbacks_list, 
                    reset_num_timesteps=True) # True as this is a new PPO model instance
        
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
        eval_env.close()
        print("Vectorized environments closed.")

if __name__ == "__main__":
    main()
