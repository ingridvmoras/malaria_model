# ODE Solver Simulation

This project implements a deterministic numerical solver for ordinary differential equation (ODE) systems, specifically designed to model the dynamics of infections and immune responses. The solver is based on the work of Zhang et al., 2023, and allows for the exploration of various parameters affecting the system.

## Project Structure

- `src/ODE_Solver_persistence.py`: Contains the core ODE solver, including the definitions of the ODEs, parameter initialization, initial conditions, and plotting functions.
- `src/simulation.py`: Implements the simulation logic to vary parameters such as the average number of infections, the infected bacteria death rate (`a`), and the maximum bacteria growth rate (`r`). It runs multiple simulations and collects results for analysis.
- `src/utils.py`: Provides utility functions for data processing, result visualization, and other helper functions used throughout the project.
- `requirements.txt`: Lists the necessary dependencies for the project, including `numpy`, `pandas`, `scipy`, `matplotlib`, and `seaborn`.

## Installation

To set up the environment, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ode-solver-simulation
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Simulations

To run the simulations, execute the `simulation.py` script:

```
python src/simulation.py
```

This will initiate the simulations with varying parameters and output the results for analysis.

## Results

The results of the simulations will be saved in a specified format (e.g., CSV files) for further analysis. You can modify the parameters in `simulation.py` to explore different scenarios.

## License

This project is licensed under the MIT License. See the LICENSE file for details.