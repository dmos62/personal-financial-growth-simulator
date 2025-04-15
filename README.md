# Financial Investment Simulator

A static web application that simulates investment growth over time using TypeScript and Apache ECharts.

## Features

- Interactive controls for adjusting simulation parameters
- Real-time visualization of investment trajectories at different percentiles (5th, 25th, 50th, 75th, 95th)
- Efficient calculation of multiple simulation trajectories
- Dynamic chart updates with smooth transitions
- Inflation-adjusted projections over a configurable time period

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm or yarn

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   npm install
   ```
   or
   ```
   yarn
   ```

### Development

Run the development server:
```
npm run start
```
or
```
yarn start
```

### Build

Build for production:
```
npm run build
```
or
```
yarn build
```

## How It Works

The application uses Geometric Brownian Motion to simulate investment returns while accounting for factors like inflation, volatility, and monthly income/expenses. Users can adjust various parameters to explore different financial scenarios.

## Technologies Used

- TypeScript
- Apache ECharts
- Vite (for building and development)
