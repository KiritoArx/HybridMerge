import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm 
import os
import math

# --- Configuration & Constants ---
COLUMN_COUNT = 5
MAX_HEIGHT = 6
BOARD_SIZE_FLAT = COLUMN_COUNT * MAX_HEIGHT  # 5 columns * 6 rows = 30
# New STATE_SIZE: board + current_tile + preview_tile
STATE_SIZE = BOARD_SIZE_FLAT + 2 
NUM_ACTIONS = COLUMN_COUNT             

EXPERT_DATA_FILE = "data/expert_games.jsonl" 
MODEL_SAVE_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "bc_policy_with_preview.pt") # New model name

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 50 
VAL_SPLIT_RATIO = 0.2 
HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 128

NORMALIZATION_FACTOR = "log2" # Or a float like 2048.0, or None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Data Loading and Preprocessing ---
class ExpertDataset(Dataset):
    def __init__(self, combined_states, actions, state_normalization_factor=None):
        self.combined_states = combined_states # This will be (board + current + preview)
        self.actions = actions
        self.state_normalization_factor = state_normalization_factor

    def __len__(self):
        return len(self.actions) # Length based on number of actions/full observations

    def __getitem__(self, idx):
        # combined_state is already pre-calculated as (board_flat, current_tile, preview_tile)
        combined_state_np = self.combined_states[idx]
        
        # Apply normalization to the entire combined state vector
        # Note: This applies the same normalization to board tiles and the current/preview tiles.
        # You might want different scaling for board vs. individual tiles if their ranges are very different.
        # For simplicity, we'll normalize the whole vector.
        if self.state_normalization_factor == "log2":
            state_float = combined_state_np.astype(np.float32)
            # Add 1 before log to handle zeros, then normalize by log2 of a potential max tile + 1
            # Using a general large value like 16384 for normalization range
            processed_state = np.log2(np.maximum(state_float, 0) + 1.0) / np.log2(16384.0 + 1.0) 
        elif isinstance(self.state_normalization_factor, (int, float)):
            processed_state = combined_state_np.astype(np.float32) / self.state_normalization_factor
        else: 
            processed_state = combined_state_np.astype(np.float32)
        
        return torch.tensor(processed_state, dtype=torch.float32), \
               torch.tensor(self.actions[idx], dtype=torch.long)

def load_and_preprocess_data(file_path, normalization_factor=None):
    print(f"Loading expert data (with current/preview tiles) from: {file_path}")
    raw_board_states = []
    current_tiles_for_decision = []
    preview_tiles_for_decision = []
    actions = []
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    data_point = json.loads(line)
                    if ('state' in data_point and 
                        'action' in data_point and
                        'current_tile_for_decision' in data_point and # <<< Check for new field
                        'preview_tile_for_decision' in data_point):   # <<< Check for new field
                        
                        if isinstance(data_point['state'], list) and len(data_point['state']) == BOARD_SIZE_FLAT:
                            raw_board_states.append(np.array(data_point['state']))
                            current_tiles_for_decision.append(data_point['current_tile_for_decision'])
                            preview_tiles_for_decision.append(data_point['preview_tile_for_decision'])
                            actions.append(data_point['action'])
                        else:
                            print(f"Warning: Skipping line {line_num+1} due to invalid state format or length.")
                    else:
                        print(f"Warning: Skipping line {line_num+1} due to missing required fields (state, action, current_tile_for_decision, preview_tile_for_decision).")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping line {line_num+1} due to JSON decode error.")
        
        if not actions: # If no actions were loaded, something is wrong
            print("Error: No valid data loaded. Check expert data file format and content.")
            return None
            
        # Combine board state, current tile, and preview tile into a single state vector
        combined_states_list = []
        for i in range(len(actions)):
            combined_state = np.concatenate((
                raw_board_states[i], 
                np.array([current_tiles_for_decision[i]]), 
                np.array([preview_tiles_for_decision[i]])
            ))
            combined_states_list.append(combined_state)
        
        print(f"Loaded and combined {len(actions)} expert transitions. New state size: {STATE_SIZE}")
        return ExpertDataset(np.array(combined_states_list), np.array(actions), normalization_factor)
    except FileNotFoundError:
        print(f"Error: Expert data file not found at {file_path}")
        return None

# BCModel and train_bc_model function remain largely the same,
# as they use STATE_SIZE which is now updated.
class BCModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(BCModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size) 
        )
    def forward(self, x):
        return self.network(x)

def train_bc_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    # (This function's internal logic is the same as before, just uses the new data loaders)
    print("Starting Behavioral Cloning training...")
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train() 
        train_loss_sum, train_correct_predictions, train_total_samples = 0,0,0
        for states, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            states, actions = states.to(DEVICE), actions.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(states) 
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * states.size(0)
            _, predicted_actions = torch.max(outputs, 1) 
            train_correct_predictions += (predicted_actions == actions).sum().item()
            train_total_samples += states.size(0)
        avg_train_loss = train_loss_sum / train_total_samples if train_total_samples > 0 else 0
        train_accuracy = train_correct_predictions / train_total_samples if train_total_samples > 0 else 0

        val_loss_sum, val_correct_predictions, val_total_samples = 0,0,0
        if val_loader: # Check if val_loader is not None
            model.eval() 
            with torch.no_grad():
                for states, actions in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                    states, actions = states.to(DEVICE), actions.to(DEVICE)
                    outputs = model(states)
                    loss = criterion(outputs, actions)
                    val_loss_sum += loss.item() * states.size(0)
                    _, predicted_actions = torch.max(outputs, 1)
                    val_correct_predictions += (predicted_actions == actions).sum().item()
                    val_total_samples += states.size(0)
        avg_val_loss = val_loss_sum / val_total_samples if val_total_samples > 0 else 0
        val_accuracy = val_correct_predictions / val_total_samples if val_total_samples > 0 else 0

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy and val_total_samples > 0 :
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with Val Acc: {best_val_accuracy:.4f}")
    
    print("Training finished.")
    if best_val_accuracy > 0 : 
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"Model saved to {model_save_path}")
    else:
        print("No model improved enough to be saved, or validation set was empty/not used.")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}") 
    print(f"Using state normalization: {NORMALIZATION_FACTOR if NORMALIZATION_FACTOR else 'None'}")

    if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)

    full_dataset = load_and_preprocess_data(EXPERT_DATA_FILE, normalization_factor=NORMALIZATION_FACTOR)

    if full_dataset and len(full_dataset) > 0 :
        val_size = int(VAL_SPLIT_RATIO * len(full_dataset))
        if val_size == 0 and len(full_dataset) > 1: val_size = 1 
        train_size = len(full_dataset) - val_size
        
        val_dataset_actual = None
        if train_size <=0 and len(full_dataset)>0:
            train_dataset = full_dataset
            print("Warning: Dataset too small for validation split. Using all data for training, no validation.")
        elif val_size == 0 :
             train_dataset = full_dataset
             print("Warning: Validation set is empty due to split configuration. Using all data for training.")
        else:
            train_dataset, val_dataset_actual = random_split(full_dataset, [train_size, val_size])

        print(f"Training set size: {len(train_dataset)}")
        if val_dataset_actual: print(f"Validation set size: {len(val_dataset_actual)}")
        else: print("Validation set size: 0")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset_actual, batch_size=BATCH_SIZE, shuffle=False) if val_dataset_actual and len(val_dataset_actual) > 0 else None

        bc_model = BCModel(STATE_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, NUM_ACTIONS).to(DEVICE)
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(bc_model.parameters(), lr=LEARNING_RATE)
        train_bc_model(bc_model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, MODEL_SAVE_PATH)
    else:
        print("Could not proceed with training due to data loading issues or empty dataset.")
