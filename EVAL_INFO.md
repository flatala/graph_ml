# Dataset Split Configuration

**NOTE:** for fixed code usages look at **`notebooks/experiments/*`**

## Timestep Ranges
- **Train**: timesteps 5-29
- **Val**: timesteps 30-33
- **Test**: timesteps 34-42

## Split Statistics

| Split | Nodes | % of Total | Illicit | Licit | Illicit Ratio |
|-------|-------|------------|---------|-------|---------------|
| Train | 109,133 | 61.75% | 7,237 | 101,896 | 6.63% |
| Val | 16,718 | 9.46% | 1,305 | 15,413 | 7.81% |
| Test | 50,876 | 28.79% | 3,218 | 47,658 | 6.33% |
| **Total** | **176,727** | **100%** | **11,760** | **164,967** | **6.65%** |

## Observation Windows to Eval
 - **observation windows:** [0, 3, 5, 7]

## Training 
  - Cohort: nodes with **t_first ≤ train_end**.
  - For each node and each **K**, feed frames **t = t_first … (t_first + K)** (this may go beyond train_end).
  - **Loss/masks at every frame t:**
    - Compute loss **only** on Train nodes with **t_first ≤ train_end**.
    - **Mask out** nodes with **t_first > train_end**).

## Validation (same for testing)
  - Cohort: nodes with **t_first ≤ val_end**.
  - For each **K**, feed frames **t = t_first … (t_first + K)**.
  - **Evaluate only** on Val nodes; **mask out** non-Val nodes and any node with **t_first > val_end** at each frame.


## The core idea
  - We are checking how extra time steps on aggregated features improve predictive power of diff model classes
    - So baselines will only benefit from extra feature aggregation. 
    - Non GCN baseline (graph-based) will benefit from updated centrality values etc at each time-step
    - GCN baseline also only benefits from updated feature aggregates
    - Weighted edge GCn baseline should benefit from better edge weights (?) after extr time + the udpated aggregates
    - Temporal models should benefit by being able to explot extra temporal structure
 - Experiments:
    - Does delaying node classification yield better accuracy?
    - Which model family 's predictive power increases the most with time?
    - Is there any advantage to using graph and temporal modelling in this scenraio?
    

    
