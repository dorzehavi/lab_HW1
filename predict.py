import xgboost as xgb
import preprocess
import numpy as np
import csv
from sys import argv

if __name__ == '__main__':
    model = xgb.Booster()
    model.load_model('xgboost_model.bin')

    path_to_test = argv[1]
    test = preprocess.load_data_to_dict(path_to_test)

    test_df = preprocess.preprocess(test)
    X_test = test_df.drop(['SepsisLabel', 'id'], axis=1)
    y_test = test_df['SepsisLabel']
    X_id = test_df['id']

    dtest = xgb.DMatrix(X_test)
    y_pred_probs = model.predict(dtest)

    # Apply threshold to convert probabilities to binary class labels
    threshold = 0.5
    y_pred = np.where(y_pred_probs >= threshold, 1, 0)

    # Store predictions in a CSV file
    predictions = zip(X_id, y_pred)
    output_file = 'prediction.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'prediction'])
        writer.writerows(predictions)

