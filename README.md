# Financial Investment Simulator

Projects investment growth using Geometric Brownian Motion and Monte Carlo simulations.

## Parameters

The simulator allows you to adjust the following parameters:

| Parameter | Description |
|-----------|-------------|
| **Initial Investment** | Starting capital for your investment portfolio in euros. This is the amount you have already saved and are investing at the beginning of the simulation. Default is 80,000. |
| **Expected Return** | The average annual return you expect from your investments before accounting for inflation. Higher values represent more optimistic growth scenarios. Default is 8%, approximating the long-term historical average return of a balanced stock/bond portfolio after management fees. |
| **Investment Volatility** | Measures the expected fluctuation in investment returns year to year. Higher volatility means more uncertainty and wider range of possible outcomes. Default is 18%, reflecting typical volatility of a portfolio with significant equity exposure while remaining conservative enough to account for diversification benefits. |
| **Monthly Net Income** | Amount you plan to contribute to your investment each month. Can be negative to simulate withdrawals from your portfolio. Default is 1,000, representing a reasonable monthly savings rate for someone with moderate to high income who is actively building wealth. |
| **Expected Inflation** | The average annual inflation rate you expect over the investment period. This affects the real purchasing power of your investment. Default is 3%, aligned with the European Central Bank's long-term inflation target plus a small buffer to account for potential policy overshoots. |
| **Inflation Volatility** | Measures the expected fluctuation in inflation rates year to year. Higher values represent more uncertainty about future inflation. Default is 2%, based on historical standard deviation of annual inflation rates in developed economies during normal economic conditions. |
| **Simulation Years** | The time horizon for your investment in years. Longer periods typically show greater compounding effects. |
| **Steps Per Year** | How many calculation steps to perform per year. Higher values provide more granular simulation but don't significantly affect results. Default is 2 steps per year, all values provide similar accuracy, but more. |

## Features

*   Visualizes uncertainty by displaying outcome ranges across percentiles (5th, 25th, 50th, 75th, 95th)
*   Accounts for inflation
*   Incorporates market and inflation volatility

## Notes

- **Default values** attempt to represent typical market conditions, but you should adjust them based on your own research and expectations
- **All results are inflation-adjusted**, showing the real purchasing power of your investment over time
- **The chart displays the 5th, 25th, 50th (median), 75th, and 95th percentiles** of possible outcomes, helping you visualize the range of potential results
- **Investment Volatility** and **Inflation Volatility** have a significant impact on the spread between percentiles - higher volatility creates wider ranges of outcomes
- **Negative monthly income** simulates regular withdrawals from your portfolio, useful for retirement planning
- **Negative returns** can be used to model bear markets or economic downturns
- **Negative inflation** represents deflation, where your money's purchasing power increases over time

## How It Works

The simulator uses Geometric Brownian Motion (GBM), a common model for stock prices, to generate possible personal worth paths. It then runs many of these simulated paths using Monte Carlo simulations (MCS) to account for randomness. Based on your inputs (initial investment, monthly contributions, return, volatility, inflation), it shows potential outcomes, visualizing the uncertainty.