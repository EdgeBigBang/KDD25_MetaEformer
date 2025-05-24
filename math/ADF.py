import pandas as pd
import statsmodels.api as sm
import numpy as np

##### Test the Stationarity of Time Series Data with One Sequence per Row
def tset_stationary_row(file_path):
    X_all = np.load(open(file_path, 'rb'), allow_pickle=True)

    data = X_all[:, :, 0]

    results_df = pd.DataFrame(columns=['ADF Statistic', 'p-value', '1%', '5%', '10%', 'is_stationary'])

    for i in range(data.shape[0]):
        series = data[i, :]
        result = sm.tsa.adfuller(series)

        # Check for stationarity
        is_stationary = result[1] < 0.05

        # Add results to DataFrame
        results_df.loc[i] = [result[0], result[1], result[4]['1%'], result[4]['5%'], result[4]['10%'], is_stationary]

    # Save results to a CSV file
    result_file_path = file_path.split('.')[-1] + 'adf_test_results.csv'
    results_df.to_csv(result_file_path, index=False)


##### Read Dataset for a Single Sequence:
def tset_stationary_single(file_path):
    data = pd.read_csv(file_path, usecols=['OT'])
    result = sm.tsa.adfuller(data)

    results_df = pd.DataFrame(columns=['ADF Statistic', 'p-value', '1%', '5%', '10%', 'is_stationary'])
    # Check for stationarity
    is_stationary = result[1] < 0.05

    # Add results to DataFrame
    results_df.loc[0] = [result[0], result[1], result[4]['1%'], result[4]['5%'], result[4]['10%'], is_stationary]
    result_file_path = file_path.split('.')[-1] + 'adf_test_results.csv'
    results_df.to_csv(result_file_path, index=False)
    print("ADF test results saved to adf_test_results.csv")


##### Read Dataset with Time Series Data in Columns
def tset_stationary_single(file_path):
    data = pd.read_csv(file_path)

    results_df = pd.DataFrame(columns=['Series Name', 'ADF Statistic', 'p-value', '1%', '5%', '10%', 'is_stationary'])

    # Perform ADF test for each column
    for column in data.columns:
        print(column)
        series = data[column]
        result = sm.tsa.adfuller(series)

        # Check for stationarity
        is_stationary = result[1] < 0.05

        # Add results to DataFrame
        results_df = results_df._append({
            'Series Name': column,
            'ADF Statistic': result[0],
            'p-value': result[1],
            '1%': result[4]['1%'],
            '5%': result[4]['5%'],
            '10%': result[4]['10%'],
            'is_stationary': is_stationary
        }, ignore_index=True)

    # Save results to a CSV file
    result_file_path = file_path.split('.')[-1] + 'adf_test_results.csv'
    results_df.to_csv(result_file_path, index=False)
    print("ADF test results saved to adf_test_results.csv")
    
if __name__ == "__main__":
    file_path = r"./dataset/ECW_switch.pkl"
    tset_stationary_row(file_path)
