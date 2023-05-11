from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")
