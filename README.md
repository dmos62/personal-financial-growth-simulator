# Financial Investment Simulator

Projects investment growth using Geometric Brownian Motion and Monte Carlo simulations.

## Features

*   Visualizes uncertainty.
*   Allows adjusting parameters like starting capital, contributions, and investment horizon.
*   Displays outcome ranges across percentiles (5th, 25th, 50th, 75th, 95th).
*   Accounts for inflation.
*   Incorporates market and inflation volatility.

## How It Works

The simulator uses Geometric Brownian Motion (GBM), a common model for stock prices, to generate possible investment paths. It then runs many of these simulated paths using Monte Carlo simulations (MCS) to account for randomness. Based on your inputs (initial investment, monthly contributions, return, volatility, inflation), it shows potential investment outcomes, visualizing the uncertainty.