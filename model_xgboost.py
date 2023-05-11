import xgboost as xgb
from sklearn.metrics import f1_score
import preprocess


if __name__ == '__main__':
    path_to_train = 'data/train/'
    path_to_test = 'data/test/'
    train = preprocess.load_data_to_dict(path_to_train)
    test = preprocess.load_data_to_dict(path_to_test)
    train_df = preprocess.preprocess(train)
    test_df = preprocess.preprocess(test)

    X_train = train_df.drop('SepsisLabel', axis=1)
    y_train = train_df['SepsisLabel']
    X_test = test_df.drop('SepsisLabel', axis=1)
    y_test = test_df['SepsisLabel']

    num_majority = sum(y_train == 0)
    num_minority = sum(y_train == 1)
    scale_pos_weight = num_majority / num_minority

    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)

    print("F1 Score: {:.2f}".format(f1))

    # Save the trained model
    model.save_model('xgboost_model.bin')
