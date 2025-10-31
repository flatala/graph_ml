import torch

EXPERIMENT_CONFIG = {
    'data_dir': '../../elliptic_dataset',
    'train_timesteps': (5, 24),  # Train cohorts: nodes appearing at t=5 to t=24
    'val_timesteps': (29, 31),    # Val cohorts: nodes appearing at t=29 to t=31 (gap ensures no overlap)
    'test_timesteps': (37, 43),   # Test cohorts: nodes appearing at t=37 to t=43 (gap ensures no overlap)
    'observation_windows': [0, 3, 5],  # K values: each node evaluated at t_first(v) + K
    'hidden_dim': 64,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 100,
    'patience': 15,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
}