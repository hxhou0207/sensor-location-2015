# README

## Overview

This is a paper implementation: [Optimal sensor deployment for 3D AOA target localization | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/7178430)

## Prerequisites

To run this project, you need to have Python and several libraries installed. Below are the steps to set up the environment.

### 2. Install Required Libraries

Use `pip` to install the required libraries. Open a terminal or command prompt and run the following command:

```sh
pip install numpy matplotlib
```

## Running the Program

To run the program, follow these steps:

1. **Open a Terminal or Command Prompt**

   **Do not use the side panel in your IDE for plots.** Instead, use the console or terminal for a better view and control. (Otherwise, you can only see the picture and not interact with it.)

2. **Navigate to the Project Directory**

   Change the directory to where `main.py` is located. 

   ```sh
   cd path/to/your/project/directory
   ```

3. **Run the Script**

   Execute the script using Python:

   ```sh
   python main.py
   ```

## Program Functionality

The program initializes sensor positions around a target and simulates the EKF process for a specified number of iterations. It includes a GUI with sliders to adjust the number of sensors, process noise (Q), sensor radius, and height. The results are visualized in a 3D scatter plot and an MSE (Mean Squared Error) plot.
