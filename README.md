ICAN: Enhanced Genetic Algorithm for LSTM Hyperparameter Optimization in Stock Market Prediction 📈

Welcome to the ICAN project repository! This project implements an advanced genetic algorithm (ICAN) to optimize the hyperparameters of an LSTM model for stock market prediction. Developed from scratch, this algorithm integrates information fusion, inter-intra crossover, and adaptive mutation to enhance prediction accuracy and efficiency.

📚 Project Overview
The ICAN algorithm was proposed in a research paper to address the challenges of hyperparameter optimization in deep learning models. This repository includes the implementation of ICAN and its application to optimize LSTM hyperparameters for predicting stock prices and trends. The key features include:

Inter-Intra Crossover: Introduces genetic diversity by combining inter-chromosome and intra-chromosome crossover operations.
Adaptive Mutation: Dynamically adjusts mutation rates based on the fitness values of chromosomes.
Stock Market Data: Utilizes real stock market data for training and validation, demonstrating practical application and effectiveness.

🚀 Getting Started
To get started with this project, follow these steps:

Clone the Repository:

git clone https://github.com/ArCNiX696/Optimizing-LSTM-Hyperparameters-for-Stock-Prediction-with-ICAN-A-Novel-GA-Approach.git

Install Dependencies:
Ensure you have the necessary libraries installed to run the project. Use the following command to install the required Python packages:

Run the ICAN Algorithm:
🛠️ Required Software and Versions
Python: 3.11.5
PyTorch: 2.1.0+cu118
CUDA: 11.7

📊 Performance Evaluation
The ICAN algorithm has been evaluated against traditional GA and other optimization methods. The results show significant improvements in prediction accuracy and execution time. Key performance metrics include Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R² score.

📁 Project Structure
LSTM_ICAN.py: Implementation of the LSTM model with ICAN hyperparameter optimization.
LSTM_GA.py: Implementation of a simple GA for comparison.
Preprocessing.py: Data preprocessing script.
plot.py: Script for visualizing results.

📈 Results and Discussion
The ICAN algorithm demonstrated superior performance in optimizing LSTM hyperparameters for stock market prediction. Key findings include:
Higher Accuracy: ICAN achieved lower error rates compared to traditional GA and other methods.
Efficiency: Reduced execution time due to effective crossover and mutation strategies.
Robustness: Consistent performance across different stock market datasets.

Happy learning and exploring! 😊📊💡
