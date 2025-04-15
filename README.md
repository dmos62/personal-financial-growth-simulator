# Financial Investment Simulator

Interactively simulates and visualizes potential investment growth using Geometric Brownian Motion and Monte Carlo simulations.

## Key Features

*   **Interactive Scenario Testing:** Adjust parameters like starting capital, contributions, and investment horizon.
*   **Probabilistic Projections:** Displays outcome ranges across percentiles (5th, 25th, 50th, 75th, 95th).
*   **Inflation Adjustment:** Accounts for inflation.
*   **Volatility Modeling:** Incorporates market and inflation volatility.

## How It Works

The simulator uses Geometric Brownian Motion (GBM), a common model for stock prices, to generate possible investment paths. It then runs many of these simulated paths using Monte Carlo simulations (MCS) to account for randomness. Based on your inputs (initial investment, monthly contributions, return, volatility), it shows potential investment outcomes.