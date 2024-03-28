# Genetic Algorithm Path Finding Visualization

## Introduction
This project implements a genetic algorithm to visualize pathfinding in a game board environment using PyQt5. The genetic algorithm generates samples (individuals) that navigate from a start point to an end point on the game board while avoiding obstacles. 

<p align="center">
[Screencast from 28-03-2024 13:23:15.webm](https://github.com/UmutDeniz26/Path_Finder_Genetic/assets/76654674/abad2e32-1111-4bd6-b7e1-a629b4274758)
</p>


## Features
- **Genetic Algorithm**: The pathfinding is achieved through a genetic algorithm that evolves a population of samples over generations.
- **Visualization**: The PyQt5 framework is used to visualize the game board, samples, obstacles, and end point.
- **Customizable Parameters**: Parameters such as learning rate, mutation rate, sample speed, and board size can be adjusted to fine-tune the pathfinding behavior.
- **Dynamic Obstacle Placement**: Users can dynamically add obstacles by clicking and dragging on the game board.

## Dependencies
- Python 3.x
- PyQt5

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/genetic-algorithm-pathfinding.git
   ```

2. Install dependencies:
   ```bash
   pip install mkl numpy PyQt5
   # See requirements.txt for detailed dependencies
   ```


## Usage
1. Run the main script:
   ```bash
   python main.py
   ```

2. Use the following controls:
   - **Left Click**: Add obstacles by clicking and dragging on the game board.
   - **Enter/Return Key**: Pause/resume the pathfinding process.
   - **Mouse Movement**: Adjust the size and position of obstacles.

3. Observe the pathfinding process as samples evolve and navigate towards the end point.

## Customization
You can customize various parameters in the `main.py` file, such as:
- Learning rate
- Mutation rate
- Sample speed
- Board size
- Generation multiplier
- Select per epoch
