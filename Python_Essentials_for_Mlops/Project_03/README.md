# Readme - Predicting Insurance Costs

This repository contains a Python script that loads health insurance data, trains a linear regression model, and evaluates the model's performance. This Readme provides an overview of the code and explains how to use it.

## Description

The `insurance_costs.py` script has the following functionalities:

1. **Data Loading**: The script loads health insurance data from a specified CSV file.

2. **Data Transformation**: It performs data transformations, including calculating the base-2 logarithm of "charges" and creating a boolean column "is_smoker" based on the "smoker" column.

3. **Model Training**: It trains a linear regression model based on the training data.

4. **Model Evaluation**: It evaluates the model's performance using the mean squared error (MSE) and the coefficient of determination (R^2) metrics.

## Usage

### Prerequisites

Ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`

### Running the Script

To run the script, follow these steps:

1. Clone this repository or download the `insurance_costs.py` file.

2. Ensure that the data file "insurance.csv" is in the same directory as the script. If you want to use a different data file, modify the `file_path` variable in the script to point to the new file.

3. Open a terminal or command prompt and navigate to the directory where the script is located.

4. Execute the script with the following command:

   ```bash
   python insurance_costs.py
   ```

5. The script will load the data, train the model, and evaluate its performance on the training and test data. The results, including MSE and R^2, will be displayed in the terminal.

### Logging Configuration

The script is configured to log errors to a log file in case any issues occur during execution. The logs are recorded in the `error.log` file.

## Customization

You can customize this script in the following ways:

- Modify data transformations in the `transform_data` method to suit your specific needs.

- Use other regression models or machine learning algorithms instead of the provided linear regression model.

- Adjust the size of the test set (e.g., `test_size` in `train_test_split`) and other model settings.

## License

This script is provided under the [MIT License](LICENSE). 

