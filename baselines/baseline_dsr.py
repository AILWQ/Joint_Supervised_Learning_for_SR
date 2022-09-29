import csv
import json
import time
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from baseline_utils import processDataFiles
from dso import DeepSymbolicRegressor

warnings.filterwarnings("ignore")


def generate_results(file_path, config_path, save_path):
    results_data = []
    test_set = processDataFiles([file_path])
    test_set = test_set.strip().split("\n")

    inference_file = open(save_path, 'w', encoding='utf-8')
    csv_writer = csv.writer(inference_file)
    csv_writer.writerow(['id', 'True_Equation', 'Predict_Equation', 'MSE', 'R^2', 'RE', 'inference_time'])

    for idx, sample in enumerate(tqdm(test_set)):
        print("===> Inferencing No.{} ...".format(idx + 1))
        t = json.loads(sample)
        X = np.array(t["X"])
        y = np.array(t["y"])
        raw_eqn = t["eq"]

        csv_data = []

        try:
            start_time = time.time()
            model = DeepSymbolicRegressor(config_path)
            model.fit(X, y)
            print(model.program_.pretty())
            equation_pred = model.program_.pretty()
            y_pred = model.predict(X)
            train_time = time.time() - start_time

            MSE = mean_squared_error(y, y_pred)
            R2 = r2_score(y, y_pred)
            RE = np.average(np.abs(y - y_pred) / y)
            predicted_tree = model.program_

            csv_data.append(idx)  # id
            csv_data.append(raw_eqn)  # True equation
            csv_data.append(equation_pred)  # predict equation
            csv_data.append(MSE)  # MSE per test
            csv_data.append(R2)
            csv_data.append(RE)
            csv_data.append(train_time)
            csv_writer.writerow(csv_data)
            print("===> Inference success！\n")

        except ValueError:
            print("===> Inference failed！\n")
            equation_pred = "NA"
            predicted_tree = "NA"
            # y_pred = "NA"
            MSE = "NA"
            R2 = "NA"
            RE = "NA"
            train_time = "NA"

        # res = {
        #     "test_index": idx,
        #     "true_equation": raw_eqn,
        #     # "true_skeleton": skeleton,
        #     "predicted_equation": equation_pred,
        #     "predicted_tree": predicted_tree,
        #     # "predicted_y": y_pred,
        #     "MSE": MSE,
        #     "R2": R2,
        #     "RE": RE,
        #     "inference_time": train_time
        # }
        # results_data.append(res)

    # results = pd.DataFrame(results_data)
    # results.to_csv(save_path, index=False)
    inference_file.close()


if __name__ == "__main__":
    data_path = "../Dataset/2_var/5000000/SSDNC/*.json"
    config_path = "./dso/config/config_regression.json"
    save_path = "./results/{}_dsr.csv".format(data_path.split('/')[-1].split('.json')[0])

    generate_results(data_path, config_path, save_path)

    # # Generate some data
    # np.random.seed(0)
    # X = np.random.rand(100, 1) * 20 - 10  # (-10, 10)
    # y = np.sin(X)
    #
    # # Create the model
    # model = DeepSymbolicRegressor(config_path)  # Alternatively, you can pass in your own config JSON path
    #
    # # Fit the model
    # model.fit(X, y)
    #
    # # View the best expression
    # print(model.program_.pretty())
    #
    # # Make predictions
    # model.predict(2 * X)
