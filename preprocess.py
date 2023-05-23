import glob
import pandas as pd
import math


def load_data_to_dict(path: str):
    patient_data = {}
    for filepath in glob.glob(path + '*.psv'):
        patient_num = int(filepath.split('/')[-1].split('_')[1].split('.')[0])

        df = pd.read_csv(filepath, sep='|')

        patient_data[patient_num] = df

    return patient_data


def normalize_minmax(series):
    return (series - series.min()) / (series.max() - series.min())


def preprocess(data: dict):
    processed_data = {}
    cols = list(data.values())[0].columns

    columns_to_remove = {'EtCO2', 'HCO3', 'FiO2', 'AST', 'Alkalinephos', 'Lactate', 'Unit1', 'Unit2'}

    special_cols = {'Age', 'Gender', 'ICULOS', 'HospAdmTime'}

    cols = [c for c in cols if c not in columns_to_remove]

    for patient_num, patient_data in data.items():
        # remove rows after sepsisLabel 1 found
        positive = False
        if 1 in patient_data['SepsisLabel'].values:
            positive = True
            idx = patient_data[patient_data['SepsisLabel'] == 1].index[0]
            patient_data = patient_data.loc[:idx]

        # imputation
        features_vals = []
        features_vals.append(patient_num)
        for c in cols[:-1]:

            if c not in special_cols:

                all_nan = patient_data[c].isna().all()

                if all_nan:
                    features_vals.extend([-1] * 10)  # Added 10 new features, so we extend by 13
                else:
                    sd_val = patient_data[c].dropna().std()
                    if math.isnan(sd_val):
                        sd_val = -1
                    mean_val = patient_data[c].dropna().mean()
                    count_non_null = patient_data[c].dropna().count()
                    count_total = len(patient_data[c])
                    features_vals.extend([mean_val, sd_val, count_non_null, count_total])

                    # First-order difference
                    diff = patient_data[c].dropna().diff()

                    diff_mean = diff.dropna().mean()
                    diff_std = diff.dropna().std()

                    if math.isnan(diff_mean):
                        diff_mean = -1
                    if math.isnan(diff_std):
                        diff_std = -1

                    features_vals.extend([diff_mean, diff_std])

                    # Last and first hour values
                    last_hour_value = patient_data[c].dropna().iloc[-1]
                    first_hour_value = patient_data[c].dropna().iloc[0]
                    features_vals.extend([last_hour_value, first_hour_value])

                    # Min/Max values
                    min_val = patient_data[c].min()
                    max_val = patient_data[c].max()
                    features_vals.extend([min_val, max_val])

            else:
                all_nan = patient_data[c].isna().all()
                if all_nan:
                    features_vals.append(-1)
                else:
                    if c == 'ICULOS':
                        features_vals.append(max(patient_data[c]))
                    else:
                        features_vals.append(patient_data[c].mean())

        if positive:
            features_vals.append(1)
        else:
            features_vals.append(0)

        processed_data[patient_num] = features_vals

    columns_new = []
    columns_new.append('id')
    for c in cols[:-1]:
        if c not in special_cols:
            columns_new.append(c)
            columns_new.append(c + '_std')
            columns_new.append(c + '_count_non_null')
            columns_new.append(c + '_count')
            columns_new.append(c + '_diffmean')
            columns_new.append(c + '_diffstd')
            columns_new.append(c + '_lasthour')
            columns_new.append(c + '_firsthour')
            columns_new.append(c + '_min')
            columns_new.append(c + '_max')

        else:
            columns_new.append(c)
    columns_new.append(cols[-1])

    processed_df = pd.DataFrame.from_dict(processed_data, orient='index', columns=columns_new)

    return processed_df
