import torch

EXPERIMENT_CONFIG = {
    'data_dir': '../../elliptic_dataset',
    'train_timesteps': (5, 26), 
    'val_timesteps': (27, 31),    
    'test_timesteps': (32, 40),
    'observation_windows': [1, 3, 5, 7],
    'hidden_dim': 128,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 100,
    'patience': 15,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
}