import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys # <<< Added for sys.path modification

# --- Add project root to sys.path ---
# This ensures that 'src.xxx' imports work correctly when running this script directly,
# or when it's imported by other scripts higher in the directory tree.
# Assumes neural_network.py is in HybridMergeAI/src/phase5_self_play/
SCRIPT_DIR_NN = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_NN = os.path.abspath(os.path.join(SCRIPT_DIR_NN, '..', '..')) # Go up two levels from phase5_self_play to HybridMergeAI
if PROJECT_ROOT_NN not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_NN)
# --- End of sys.path modification ---

# Import constants from your existing BC training script
try:
    from src.bc.train_bc import STATE_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, NUM_ACTIONS, BCModel
    print(f"Successfully imported constants from src.bc.train_bc: STATE_SIZE={STATE_SIZE}")
except ImportError as e:
    print(f"Warning: Could not import constants from src.bc.train_bc: {e}. Using fallback values.")
    print("Ensure src.bc.train_bc exists and PROJECT_ROOT is added to sys.path correctly.")
    STATE_SIZE = 32  # board (5x6=30) + current_tile (1) + preview_tile (1)
    HIDDEN_SIZE_1 = 256
    HIDDEN_SIZE_2 = 128
    NUM_ACTIONS = 5  # Number of columns
    # Define a dummy BCModel if the import fails, so load_bc_weights_for_policy doesn't crash immediately
    # This is mainly for allowing the script to run; the real BCModel structure is crucial.
    class BCModel(nn.Module): # Dummy definition
        def __init__(self, i, h1, h2, o):
            super(BCModel, self).__init__()
            self.network = nn.Sequential(nn.Linear(i,h1), nn.ReLU(), nn.Linear(h1,h2), nn.ReLU(), nn.Linear(h2,o))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyValueNetwork(nn.Module):
    """
    Neural Network for AlphaZero-like self-play.
    Outputs both a policy (action probabilities) and a value (expected game outcome).
    """
    def __init__(self, input_size=STATE_SIZE, hidden_size1=HIDDEN_SIZE_1, 
                 hidden_size2=HIDDEN_SIZE_2, num_actions=NUM_ACTIONS):
        super(PolicyValueNetwork, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        # Shared body (can be same as BC model's feature extractor)
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # Policy head
        self.policy_head = nn.Linear(hidden_size2, num_actions)

        # Value head
        self.value_head = nn.Linear(hidden_size2, 1) # Outputs a single scalar value

        print(f"PolicyValueNetwork initialized. Input: {input_size}, Actions: {num_actions}, Hidden: {hidden_size1}, {hidden_size2}")

    def forward(self, x):
        # Ensure input is float
        if x.dtype != torch.float32:
            x = x.float()
            
        # Shared body
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Policy head output (logits)
        policy_logits = self.policy_head(x)
        
        # Value head output (scalar value)
        value = torch.tanh(self.value_head(x)) 
        
        return policy_logits, value

    def load_bc_weights_for_policy(self, bc_model_path):
        """
        Loads weights from a trained BCModel into the shared body and policy head.
        The value head remains randomly initialized or uses its default initialization.
        """
        if not os.path.exists(bc_model_path):
            print(f"Warning: BC model file not found at {bc_model_path}. Cannot load BC weights.")
            return False

        try:
            temp_bc_model = BCModel(self.input_size, 
                                    self.fc1.out_features, 
                                    self.fc2.out_features, 
                                    self.num_actions).to(DEVICE) 
            
            temp_bc_model.load_state_dict(torch.load(bc_model_path, map_location=DEVICE))
            print(f"Successfully loaded BC model weights from {bc_model_path} for weight transfer.")

            bc_sequential_network = temp_bc_model.network

            self.fc1.weight.data.copy_(bc_sequential_network[0].weight.data)
            self.fc1.bias.data.copy_(bc_sequential_network[0].bias.data)
            self.fc2.weight.data.copy_(bc_sequential_network[2].weight.data)
            self.fc2.bias.data.copy_(bc_sequential_network[2].bias.data)
            self.policy_head.weight.data.copy_(bc_sequential_network[4].weight.data)
            self.policy_head.bias.data.copy_(bc_sequential_network[4].bias.data)
            
            print("Successfully transferred weights from BC policy to PolicyValueNetwork (shared body & policy head).")
            print("Value head remains with its initial random weights.")
            return True
        except Exception as e:
            print(f"ERROR: Could not load or transfer BC weights: {e}")
            return False

if __name__ == '__main__':
    print(f"\n--- Running PolicyValueNetwork Test ---")
    print(f"Device for test: {DEVICE}")
    # Ensure PROJECT_ROOT_NN is used for constructing paths if needed
    # (it's defined at the top of the script)
    
    net = PolicyValueNetwork().to(DEVICE) # Uses constants imported or fallback
    
    dummy_state = torch.randn(1, STATE_SIZE).to(DEVICE) # STATE_SIZE from import or fallback
    print(f"Dummy state shape: {dummy_state.shape}")

    policy_logits, value = net(dummy_state)
    
    print("\nRaw Policy Logits (batch_size, num_actions):")
    print(policy_logits)
    print(f"Policy Logits shape: {policy_logits.shape}")

    policy_probs = F.softmax(policy_logits, dim=1)
    print("\nPolicy Probabilities (after softmax):")
    print(policy_probs)
    print(f"Policy Probs shape: {policy_probs.shape}")
    
    print("\nValue Output (batch_size, 1):")
    print(value)
    print(f"Value shape: {value.shape}")

    # Construct absolute path for BC_MODEL_PATH_FOR_TEST using PROJECT_ROOT_NN
    # Assumes 'models' directory is directly under PROJECT_ROOT_NN
    BC_MODEL_PATH_FOR_TEST = os.path.join(PROJECT_ROOT_NN, "models", "bc_policy_with_preview.pt") 
                                                                                 
    print(f"\nAttempting to load BC weights from (absolute path): {BC_MODEL_PATH_FOR_TEST}")
    if os.path.exists(BC_MODEL_PATH_FOR_TEST):
        fc1_weight_before = net.fc1.weight.data.clone()
        loaded_successfully = net.load_bc_weights_for_policy(BC_MODEL_PATH_FOR_TEST)
        
        if loaded_successfully and not torch.equal(fc1_weight_before, net.fc1.weight.data):
            print("FC1 weights changed after loading BC policy, which is expected.")
        elif loaded_successfully:
            print("BC weights loaded, but FC1 weights seem unchanged (this might be okay if they were already similar).")
        
        policy_logits_after_bc, value_after_bc = net(dummy_state)
        print("\nPolicy Logits after attempting BC weight load:")
        print(policy_logits_after_bc)
        print("\nValue Output after attempting BC weight load (value head should be unchanged by this method):")
        print(value_after_bc)
    else:
        print(f"Skipping BC weight loading test: File not found at {BC_MODEL_PATH_FOR_TEST}")
    print(f"--- End of PolicyValueNetwork Test ---")
