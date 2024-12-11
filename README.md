# DeFi Pool k-means Clustering & SD Score Prediction

## Project Overview

This project focuses on two main tasks:
1. Clustering DeFi pools using k-means clustering.
2. Analyzing and predicting SD Scores and forward-looking scores using various machine learning models.

## Files

- `DeFi_Pool_k-means_clustering.ipynb`: Notebook for clustering DeFi pools using k-means clustering.
- `SD_Score_Prediction.ipynb`: Notebook for analyzing and predicting SD Scores and forward-looking scores.

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/RazDwa1923/DeFi_Pool_k-means_clustering.git
   ```
2. Navigate to the project directory:
   ```sh
   cd DeFi_Pool_k-means_clustering
   ```
3. Install the required libraries:
   ```sh
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

## Usage

### Clustering DeFi Pools

1. Open the `DeFi_Pool_k-means_clustering.ipynb` notebook.
2. Run the cells to perform k-means clustering on DeFi pools.
3. Change the cluster number in the code to view different clusters.

### SD Score Prediction

1. Open the `SD_Score_Prediction.ipynb` notebook.
2. Run the cells to analyze and visualize the data.
3. Use the provided machine learning models to predict SD Scores and forward-looking scores.

## Machine Learning Models

The following models are used for prediction:

- Linear Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Decision Tree Classifier
- Random Forest Classifier
- Voting Classifier
- AdaBoost Classifier

## Data

- `Datasets/DeFi_Quant_Data.csv`: Contains DeFi quantitative data.
- `Datasets/asset_prices.csv`: Contains asset prices data.

## Results

The results of the clustering and prediction models are displayed in the respective notebooks. The performance metrics such as mean squared error, r2 score, accuracy, precision, recall, and f1 score are calculated and displayed.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- The data used in this project is from the DeFi Quant Data set.
- Special thanks to the open-source community for providing the necessary libraries and tools.
