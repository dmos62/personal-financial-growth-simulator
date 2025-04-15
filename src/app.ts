// Import ECharts
import * as echarts from 'echarts';

// Type definitions
interface SimulationConfig {
    invested: number;
    investmentExpectedReturn: number;
    investmentVolatility: number;
    netIncomeMonthly: number;
    inflationExpected: number;
    inflationVolatility: number;
    totalYears: number;
    stepsPerYear: number;
}

type PercentileKey = 'p5' | 'p25' | 'p50' | 'p75' | 'p95';
type PercentileData = Record<PercentileKey, number[]>;

// Constants
const PERCENTILES = [5, 25, 50, 75, 95];
const PERCENTILE_KEYS: PercentileKey[] = ['p5', 'p25', 'p50', 'p75', 'p95'];
const PERCENTILE_COLORS = {
    p5: '#91cc75',
    p25: '#5470c6',
    p50: '#ee6666',
    p75: '#5470c6',
    p95: '#91cc75'
};

const OPTIMAL_TRAJECTORY_COUNT = 1000;

// Default configuration
const DEFAULT_CONFIG: SimulationConfig = {
    invested: 80000,
    investmentExpectedReturn: 0.08,
    investmentVolatility: 0.18,
    netIncomeMonthly: 1000,
    inflationExpected: 0.03,
    inflationVolatility: 0.02,
    totalYears: 10,
    stepsPerYear: 2
};

// Current configuration (will be updated by UI)
let currentConfig: SimulationConfig = { ...DEFAULT_CONFIG };

// Chart instance
let chart: echarts.ECharts | null = null;

// Simulation functions
function simulateGBM(invested: number, expectedReturn: number, volatility: number, stepsInYear: number): number {
    const timeFraction = 1 / stepsInYear;
    const drift = (expectedReturn - 0.5 * volatility ** 2) * timeFraction;
    const volatilityStep = volatility * Math.sqrt(timeFraction);
    const randomShock = volatilityStep * randomNormal();
    const growthFactor = Math.exp(drift + randomShock);
    return invested * growthFactor;
}

function stepFinancialGrowth(
    invested: number,
    investmentExpectedReturn: number,
    investmentVolatility: number,
    netIncomePerStep: number,
    inflationExpected: number,
    inflationVolatility: number,
    stepsInYear: number
): [number, number] {
    // Simulate investment growth
    const newInvested = simulateGBM(invested, investmentExpectedReturn, investmentVolatility, stepsInYear);

    // Simulate inflation impact
    const inflationFactor = simulateGBM(1, inflationExpected, inflationVolatility, stepsInYear);

    const newNetIncomePerStep = netIncomePerStep * inflationFactor;

    // Adjust for income
    let adjustedInvested = newInvested + newNetIncomePerStep;

    // Adjust for inflation impact (reduce real value of investment)
    adjustedInvested /= inflationFactor;

    return [adjustedInvested, newNetIncomePerStep];
}

function simulatePeriodOfGrowth(config: SimulationConfig): number[] {
    const {
        invested,
        investmentExpectedReturn,
        investmentVolatility,
        netIncomeMonthly,
        inflationExpected,
        inflationVolatility,
        totalYears,
        stepsPerYear
    } = config;

    const perTimeStep: number[] = [invested];
    const totalSteps = totalYears * stepsPerYear;

    const netIncomeYearly = netIncomeMonthly * 12;
    const netIncomePerStep = netIncomeYearly / stepsPerYear;

    let currentInvested = invested;
    let currentNetIncome = netIncomePerStep;

    for (let i = 0; i < totalSteps; i++) {
        [currentInvested, currentNetIncome] = stepFinancialGrowth(
            currentInvested,
            investmentExpectedReturn,
            investmentVolatility,
            currentNetIncome,
            inflationExpected,
            inflationVolatility,
            stepsPerYear
        );
        perTimeStep.push(currentInvested);
    }

    return perTimeStep;
}

// Helper functions
function randomNormal(): number {
    // Box-Muller transform for normal distribution
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function generateTimeLabels(totalYears: number, stepsPerYear: number): string[] {
    const totalSteps = totalYears * stepsPerYear + 1; // +1 for initial value
    const labels: string[] = [];

    const today = new Date();
    const startYear = today.getFullYear();
    const startMonth = today.getMonth();

    for (let i = 0; i < totalSteps; i++) {
        const stepInYears = i / stepsPerYear;
        const yearOffset = Math.floor(stepInYears);
        const monthOffset = Math.floor((stepInYears - yearOffset) * 12);

        const year = startYear + yearOffset;
        const month = (startMonth + monthOffset) % 12;

        if (stepsPerYear <= 4) {
            // For quarterly or less frequent steps, just show the year
            labels.push(`${year}`);
        } else {
            // For more frequent steps, show month and year
            const date = new Date(year, month);
            labels.push(date.toLocaleDateString(undefined, { year: 'numeric', month: 'short' }));
        }
    }

    return labels;
}

function calculatePercentiles(trajectories: number[][]): PercentileData {
    const result: Partial<PercentileData> = {};
    const timeSteps = trajectories[0].length;

    // Initialize arrays for each percentile
    PERCENTILE_KEYS.forEach(key => {
        result[key] = new Array(timeSteps).fill(0);
    });

    // Calculate percentiles for each time step
    for (let timeStep = 0; timeStep < timeSteps; timeStep++) {
        // Extract values for this time step across all trajectories
        const valuesAtTimeStep = trajectories.map(traj => traj[timeStep]);

        // Sort values for percentile calculation
        valuesAtTimeStep.sort((a, b) => a - b);

        // Calculate each percentile
        for (let i = 0; i < PERCENTILES.length; i++) {
            const percentile = PERCENTILES[i];
            const key = PERCENTILE_KEYS[i];
            const index = Math.floor(valuesAtTimeStep.length * percentile / 100);
            result[key]![timeStep] = valuesAtTimeStep[index];
        }
    }

    return result as PercentileData;
}

// Chart functions
function initializeChart(): void {
    const chartDom = document.getElementById('chart')!;
    chart = echarts.init(chartDom);

    // Set initial options
    const options: echarts.EChartsOption = {
        title: {
            text: 'Investment Growth Over Time (Inflation Adjusted)',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            formatter: function(params: any) {
                const timeLabel = params[0].axisValue;
                let result = `<div style="font-weight:bold;margin-bottom:5px;">${timeLabel}</div>`;

                params.forEach((param: any) => {
                    const value = param.value;
                    const formattedValue = new Intl.NumberFormat('en-US', {
                        style: 'currency',
                        currency: 'EUR',
                        maximumFractionDigits: 0
                    }).format(value);

                    const percentile = param.seriesName.replace('p', '');
                    result += `<div style="display:flex;justify-content:space-between;align-items:center;margin:3px 0;">
                        <span style="display:inline-block;margin-right:5px;border-radius:50%;width:10px;height:10px;background-color:${param.color};"></span>
                        <span style="flex:1;">${percentile}th Percentile:</span>
                        <span style="font-weight:bold;">${formattedValue}</span>
                    </div>`;
                });

                return result;
            }
        },
        legend: {
            data: PERCENTILE_KEYS.map(key => `${key.replace('p', '')}th Percentile`),
            bottom: 10
        },
        grid: {
            left: '5%',
            right: '5%',
            bottom: '15%',
            top: '15%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: [],
            axisLabel: {
                rotate: 45
            }
        },
        yAxis: {
            type: 'value',
            axisLabel: {
                formatter: (value: number) => {
                    if (value >= 1000000) {
                        return `€${(value / 1000000).toFixed(1)}M`;
                    } else if (value >= 1000) {
                        return `€${(value / 1000).toFixed(0)}K`;
                    } else {
                        return `€${value.toFixed(0)}`;
                    }
                }
            }
        },
        series: PERCENTILE_KEYS.map(key => ({
            name: `${key.replace('p', '')}th Percentile`,
            type: 'line',
            data: [],
            smooth: true,
            lineStyle: {
                width: key === 'p50' ? 3 : 2
            },
            itemStyle: {
                color: PERCENTILE_COLORS[key]
            },
            emphasis: {
                focus: 'series'
            }
        }))
    };

    chart.setOption(options as any);
}

function updateChart(percentileData: PercentileData, timeLabels: string[]): void {
    if (!chart) return;

    const seriesData = PERCENTILE_KEYS.map(key => ({
        name: `${key.replace('p', '')}th Percentile`,
        data: percentileData[key]
    }));

    chart.setOption({
        xAxis: {
            data: timeLabels
        },
        series: seriesData
    } as any);
}

// Simulation runner
async function runSimulation(): Promise<void> {
    // Generate multiple trajectories
    const trajectories: number[][] = [];

    // Generate initial trajectories
    for (let i = 0; i < 50; i++) {
        trajectories.push(simulatePeriodOfGrowth(currentConfig));
    }

    // Calculate percentiles and update chart
    const percentileData = calculatePercentiles(trajectories);
    const timeLabels = generateTimeLabels(currentConfig.totalYears, currentConfig.stepsPerYear);
    updateChart(percentileData, timeLabels);

    // Continue generating more trajectories in the background
    setTimeout(async () => {
        for (let i = 0; i < OPTIMAL_TRAJECTORY_COUNT - 50; i += 10) {
            // Generate in batches of 10
            const newTrajectories = [];
            for (let j = 0; j < 10 && i + j < OPTIMAL_TRAJECTORY_COUNT - 50; j++) {
                newTrajectories.push(simulatePeriodOfGrowth(currentConfig));
            }

            trajectories.push(...newTrajectories);

            // Update chart every 100 trajectories
            if (trajectories.length % 100 === 0 || trajectories.length >= OPTIMAL_TRAJECTORY_COUNT) {
                const updatedPercentileData = calculatePercentiles(trajectories);
                updateChart(updatedPercentileData, timeLabels);

                // Allow UI to update
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
    }, 0);
}

// UI event handlers
function setupEventListeners(): void {
    // Set up slider and input synchronization
    const parameterIds = [
        'invested',
        'investment-expected-return',
        'investment-volatility',
        'net-income-monthly',
        'inflation-expected',
        'inflation-volatility',
        'total-years',
        'steps-per-year'
    ];

    parameterIds.forEach(id => {
        const slider = document.getElementById(`${id}-slider`) as HTMLInputElement;
        const input = document.getElementById(`${id}-input`) as HTMLInputElement;

        // Sync slider to input
        slider.addEventListener('input', () => {
            let value = parseFloat(slider.value);

            // Convert percentage values
            if (['investment-expected-return', 'investment-volatility', 'inflation-expected', 'inflation-volatility'].includes(id)) {
                input.value = value.toString();
                value = value / 100; // Convert from percentage to decimal
            } else {
                input.value = value.toString();
            }

            updateConfigFromUI();
        });

        // Sync input to slider (only if within slider bounds)
        input.addEventListener('input', () => {
            let value = parseFloat(input.value);

            if (!isNaN(value)) {
                // Get slider bounds
                const min = parseFloat(slider.min);
                const max = parseFloat(slider.max);

                // Check if value is outside slider bounds
                const isOutOfRange = value < min || value > max;

                // Add or remove visual indicator class
                if (isOutOfRange) {
                    input.classList.add('out-of-range');
                } else {
                    input.classList.remove('out-of-range');
                    // Only update slider if value is within bounds
                    slider.value = value.toString();
                }

                // Convert percentage values if needed
                if (['investment-expected-return', 'investment-volatility', 'inflation-expected', 'inflation-volatility'].includes(id)) {
                    value = value / 100; // Convert from percentage to decimal
                }

                updateConfigFromUI();
            }
        });
    });

    // Auto update checkbox
    const autoUpdateCheckbox = document.getElementById('auto-update') as HTMLInputElement;
    const updateButton = document.getElementById('update-button') as HTMLButtonElement;

    autoUpdateCheckbox.addEventListener('change', () => {
        updateButton.disabled = autoUpdateCheckbox.checked;
    });

    // Update button
    updateButton.addEventListener('click', () => {
        runSimulation();
    });
}

function updateConfigFromUI(): void {
    const newConfig: SimulationConfig = {
        invested: parseFloat((document.getElementById('invested-input') as HTMLInputElement).value),
        investmentExpectedReturn: parseFloat((document.getElementById('investment-expected-return-input') as HTMLInputElement).value) / 100,
        investmentVolatility: parseFloat((document.getElementById('investment-volatility-input') as HTMLInputElement).value) / 100,
        netIncomeMonthly: parseFloat((document.getElementById('net-income-monthly-input') as HTMLInputElement).value),
        inflationExpected: parseFloat((document.getElementById('inflation-expected-input') as HTMLInputElement).value) / 100,
        inflationVolatility: parseFloat((document.getElementById('inflation-volatility-input') as HTMLInputElement).value) / 100,
        totalYears: parseFloat((document.getElementById('total-years-input') as HTMLInputElement).value),
        stepsPerYear: parseFloat((document.getElementById('steps-per-year-input') as HTMLInputElement).value)
    };

    // Update current config
    currentConfig = newConfig;

    // Run simulation if auto-update is enabled
    if ((document.getElementById('auto-update') as HTMLInputElement).checked) {
        runSimulation();
    }
}

// Check if input values are outside slider bounds and apply visual indicator
function checkInputRanges(): void {
    const parameterIds = [
        'invested',
        'investment-expected-return',
        'investment-volatility',
        'net-income-monthly',
        'inflation-expected',
        'inflation-volatility',
        'total-years',
        'steps-per-year'
    ];

    parameterIds.forEach(id => {
        const slider = document.getElementById(`${id}-slider`) as HTMLInputElement;
        const input = document.getElementById(`${id}-input`) as HTMLInputElement;

        const value = parseFloat(input.value);
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);

        if (value < min || value > max) {
            input.classList.add('out-of-range');
        } else {
            input.classList.remove('out-of-range');
        }
    });
}

// Initialize the application
function initializeApp(): void {
    // Initialize chart
    initializeChart();

    // Set up event listeners
    setupEventListeners();

    // Check initial input ranges
    checkInputRanges();

    // Run initial simulation
    runSimulation();

    // Handle window resize
    window.addEventListener('resize', () => {
        if (chart) {
            chart.resize();
        }
    });
}

// Start the application when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeApp);
