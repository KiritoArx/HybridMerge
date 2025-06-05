import numpy as np
import math
# import copy # No longer needed for deepcopying MergeEnv
import torch
import torch.nn.functional as F
import os
import sys

# --- Add project root to sys.path for imports ---
SCRIPT_DIR_MCTS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_MCTS = os.path.abspath(os.path.join(SCRIPT_DIR_MCTS, '..', '..'))
if PROJECT_ROOT_MCTS not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_MCTS)
# --- End of sys.path modification ---

from src.env.merge_env import MergeEnv 
from src.phase5_self_play.neural_network import PolicyValueNetwork, DEVICE 
try:
    from src.bc.train_bc import NUM_ACTIONS # If available
except ImportError:
    NUM_ACTIONS = 5 # Fallback
    print(f"Warning: NUM_ACTIONS not imported from train_bc, using fallback: {NUM_ACTIONS}")


# MCTS Configuration Constants
CPUCT = 1.0  
DIRICHLET_ALPHA = 0.3 
DIRICHLET_EPSILON = 0.25 

class Node: # (Node class remains the same as before - ID: phase5_mcts_py)
    def __init__(self, parent=None, prior_p=0.0, action_taken=None):
        self.parent = parent
        self.children = {} 
        self.action_taken = action_taken 
        self.N = 0  
        self.W = 0.0  
        self.Q = 0.0  
        self.P = prior_p  

    def is_leaf(self):
        return len(self.children) == 0
    def is_root(self):
        return self.parent is None
    def select_child(self, c_puct=CPUCT):
        best_score = -float('inf')
        best_action = -1
        best_child = None
        for action, child_node in self.children.items():
            ucb_score = child_node.Q + \
                        c_puct * child_node.P * \
                        (math.sqrt(self.N) / (1 + child_node.N))
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child_node
        if best_child is None and self.children: 
             best_action = list(self.children.keys())[0]
             best_child = self.children[best_action]
        elif not self.children: return None, None
        return best_action, best_child
    def expand(self, legal_actions, action_probs):
        for action in legal_actions:
            if action not in self.children: 
                self.children[action] = Node(parent=self, prior_p=action_probs[action], action_taken=action)
    def update(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N if self.N > 0 else 0.0
    def backpropagate(self, value):
        self.update(value)
        if self.parent:
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, policy_value_network, game_env_class_for_simulation, num_simulations, device, num_actions):
        self.network = policy_value_network.to(device)
        self.network.eval() 
        self.game_env_class_for_simulation = game_env_class_for_simulation # e.g., MergeEnv class
        self.num_simulations = num_simulations
        self.device = device
        self.num_actions = num_actions
        self.root = None

    def _get_game_state_for_nn(self, mcts_sim_env_instance):
        obs_np = mcts_sim_env_instance._get_obs() 
        return torch.tensor(obs_np.astype(np.float32)).unsqueeze(0).to(self.device)

    def _get_legal_actions(self, mcts_sim_env_instance):
        legal_actions = []
        for col in range(mcts_sim_env_instance.column_count):
            if not mcts_sim_env_instance.game_engine.is_column_full(col): # Uses the C++ method
                legal_actions.append(col)
        return legal_actions

    def run_simulation(self, current_sim_node, mcts_sim_env_instance):
        node = current_sim_node
        path = [node] 
        
        while not node.is_leaf():
            action, next_node = node.select_child()
            if next_node is None: break 
            _, _, terminated, _, _ = mcts_sim_env_instance.step(action) # Apply action to this sim's env
            node = next_node
            path.append(node)
            if terminated: 
                value = -1.0 # Game ended, assign terminal value (e.g., loss)
                for visited_node in reversed(path):
                    visited_node.backpropagate(value)
                return 

        if mcts_sim_env_instance.game_engine.is_game_over():
            value = -1.0 
            for visited_node in reversed(path):
                 visited_node.backpropagate(value)
            return

        obs_for_nn = self._get_game_state_for_nn(mcts_sim_env_instance)
        with torch.no_grad():
            policy_logits, value_estimate_tensor = self.network(obs_for_nn)
        
        policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value_estimate_tensor.item() 

        legal_actions = self._get_legal_actions(mcts_sim_env_instance)
        if not legal_actions: 
            value = -1.0 
            for visited_node in reversed(path):
                 visited_node.backpropagate(value)
            return

        masked_policy_probs = np.zeros_like(policy_probs)
        if legal_actions: 
            for legal_action in legal_actions:
                masked_policy_probs[legal_action] = policy_probs[legal_action]
            if np.sum(masked_policy_probs) > 1e-6: 
                masked_policy_probs /= np.sum(masked_policy_probs)
            else: 
                for legal_action in legal_actions:
                    masked_policy_probs[legal_action] = 1.0 / len(legal_actions)
        
        node.expand(legal_actions, masked_policy_probs) 
        for visited_node in reversed(path):
            visited_node.backpropagate(value)

    def get_action_probabilities(self, root_game_env_instance, temperature=1.0):
        # Get initial policy for the root state from the network
        root_obs_for_nn = self._get_game_state_for_nn(root_game_env_instance)
        with torch.no_grad():
            policy_logits_root, _ = self.network(root_obs_for_nn)
        policy_probs_root = F.softmax(policy_logits_root, dim=1).squeeze(0).cpu().numpy()

        root_legal_actions = self._get_legal_actions(root_game_env_instance)
        if not root_legal_actions:
            # print("Warning: No legal actions from root state in get_action_probabilities.")
            return np.ones(self.num_actions) / self.num_actions # Fallback

        masked_policy_probs_root = np.zeros_like(policy_probs_root)
        for action in root_legal_actions:
            masked_policy_probs_root[action] = policy_probs_root[action]
        
        sum_masked_probs = np.sum(masked_policy_probs_root)
        if sum_masked_probs > 1e-6:
             final_policy_probs_root_for_expansion = masked_policy_probs_root / sum_masked_probs
        else: 
            final_policy_probs_root_for_expansion = np.zeros_like(masked_policy_probs_root)
            for action in root_legal_actions:
                final_policy_probs_root_for_expansion[action] = 1.0 / len(root_legal_actions)

        if DIRICHLET_EPSILON > 0 and len(root_legal_actions) > 0:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root_legal_actions))
            noisy_policy_root = np.copy(final_policy_probs_root_for_expansion) # Operate on a copy
            
            noise_idx = 0
            for action_idx in range(len(noisy_policy_root)):
                if action_idx in root_legal_actions: # Only apply noise to legal actions' probs
                    noisy_policy_root[action_idx] = (1 - DIRICHLET_EPSILON) * noisy_policy_root[action_idx] + DIRICHLET_EPSILON * noise[noise_idx]
                    noise_idx +=1
            
            # Re-normalize if necessary (though weighted sum of two normalized distributions might not need it if weights sum to 1)
            if np.sum(noisy_policy_root) > 1e-6: noisy_policy_root /= np.sum(noisy_policy_root)
            final_policy_probs_root_for_expansion = noisy_policy_root


        self.root = Node(prior_p=1.0) 
        self.root.expand(root_legal_actions, final_policy_probs_root_for_expansion)

        for _ in range(self.num_simulations):
            # --- Create a new env and set its state for each simulation ---
            mcts_sim_env = self.game_env_class_for_simulation(render_mode=None, shaping_strength=0.0) # Assuming constructor defaults
            
            # Get serializable state from the true root environment
            root_env_state_data = root_game_env_instance.get_serializable_game_state()
            
            # Set the simulation environment to this root state
            mcts_sim_env.set_serializable_game_state(root_env_state_data)
            # --- End of new state setting block ---
            
            self.run_simulation(self.root, mcts_sim_env)

        visit_counts = np.zeros(self.num_actions, dtype=float)
        for action_idx in range(self.num_actions):
            if action_idx in self.root.children:
                visit_counts[action_idx] = self.root.children[action_idx].N
        
        if np.sum(visit_counts) == 0: 
            return final_policy_probs_root_for_expansion # Fallback if no visits

        if temperature == 0: 
            action_probs_final = np.zeros_like(visit_counts, dtype=float)
            best_action = -1
            max_visits = -1
            for action_idx in root_legal_actions: 
                if visit_counts[action_idx] > max_visits:
                    max_visits = visit_counts[action_idx]
                    best_action = action_idx
            if best_action == -1 and root_legal_actions: best_action = root_legal_actions[0]
            if best_action != -1: action_probs_final[best_action] = 1.0
            return action_probs_final
        else:
            adjusted_counts = visit_counts ** (1.0 / temperature)
            sum_adjusted_counts = np.sum(adjusted_counts)
            if sum_adjusted_counts > 1e-6:
                action_probs_final = adjusted_counts / sum_adjusted_counts
            else: 
                action_probs_final = np.zeros(self.num_actions, dtype=float)
                if root_legal_actions:
                    for la in root_legal_actions: action_probs_final[la] = 1.0 / len(root_legal_actions)
                else: return np.ones(self.num_actions) / self.num_actions
            return action_probs_final

if __name__ == '__main__':    
    print("MCTS Script (Using Manual State Setting for Simulations)")
    print("To test: Ensure MergeEnv has get/set_serializable_game_state and _merge_engine has set_full_game_state_from_flat.")
    print("Ensure C++ Game class has 'is_column_full(col_idx)' exposed via Pybind11.")

    print("\n--- MCTS Basic Functional Test (with manual state setting) ---")
    try:
        try:
            from src.bc.train_bc import STATE_SIZE as BC_STATE_SIZE, NUM_ACTIONS as BC_NUM_ACTIONS
        except:
            BC_STATE_SIZE = 32; BC_NUM_ACTIONS = 5
            print("Using fallback STATE_SIZE and NUM_ACTIONS for MCTS test.")

        root_game_env = MergeEnv(render_mode=None, shaping_strength=0.0)
        initial_obs, info = root_game_env.reset()
        print(f"Initial observation for MCTS test: {initial_obs.shape}")

        pv_network = PolicyValueNetwork(input_size=BC_STATE_SIZE, num_actions=BC_NUM_ACTIONS).to(DEVICE)
        
        bc_model_path_for_mcts_test = os.path.join(PROJECT_ROOT_MCTS, "models", "bc_policy_with_preview.pt")
        print(f"Attempting to load BC weights for MCTS test from: {bc_model_path_for_mcts_test}")
        if os.path.exists(bc_model_path_for_mcts_test):
            pv_network.load_bc_weights_for_policy(bc_model_path_for_mcts_test)
        else:
            print(f"BC model for MCTS test not found. Using randomly initialized policy head.")
        
        mcts_searcher = MCTS(policy_value_network=pv_network, 
                               game_env_class_for_simulation=MergeEnv, # Pass the class
                               num_simulations=50, 
                               device=DEVICE,
                               num_actions=BC_NUM_ACTIONS)
        
        print("MCTS instance created.")
        print("Running MCTS to get action probabilities for initial state...")
        action_probs = mcts_searcher.get_action_probabilities(root_game_env, temperature=1.0)
        
        print(f"\nMCTS Action Probabilities for initial state: {action_probs}")
        if action_probs is not None and len(action_probs) > 0 and np.sum(action_probs) > 0: # Check if probs are valid
            chosen_action = np.random.choice(len(action_probs), p=action_probs)
            print(f"Example action chosen based on MCTS probs: {chosen_action}")
            next_obs, reward, terminated, truncated, info = root_game_env.step(chosen_action)
            print(f"Root game env stepped with action {chosen_action}. Terminated: {terminated}. Intrinsic Score: {info.get('intrinsic_score_this_step')}")
        else:
            print("MCTS did not return valid action probabilities or no legal moves.")
            
    except ImportError as e:
        print(f"Could not run MCTS test due to import error: {e}")
    except AttributeError as e:
        print(f"AttributeError during MCTS test: {e}")
    except Exception as e:
        print(f"An error occurred during MCTS test: {e}")
        import traceback
        traceback.print_exc()

    print("--- MCTS Test Block Finished ---")

