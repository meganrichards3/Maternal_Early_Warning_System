import random
from typing import List
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df


def optimize_dtypes(df: pd.DataFrame, datetime_features: List[str] = []):
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))


def find_label_max(dataset, label_column_title):
    if np.max(dataset[label_column_title]) == 1.0:
        return 1
    else:
        return 0


def sample_for_class_balance(data, label_column, prevalence, encounter_col):
    label_max = data.groupby(encounter_col).apply(find_label_max, label_column)
    pos_ids = list(label_max[label_max == 1].index)
    neg_ids = list(label_max[label_max == 0].index)

    pos_set = data[data[encounter_col].isin(pos_ids)]
    neg_set = data[data[encounter_col].isin(neg_ids)]

    num_positive = pos_set.reset_index()[encounter_col].nunique()
    num_negative = min(int(num_positive / prevalence - num_positive), int(label_max.size - len(pos_ids)))

    negative_ids = neg_set.reset_index()[encounter_col].drop_duplicates().sample(num_negative)
    negative_sampled = data[data[encounter_col].isin(negative_ids)]

    return pd.concat([pos_set, negative_sampled])


def encode_categorical_variables(df, categorical_cols):
    for col in categorical_cols:
        if col in df.columns.values:
            enc = LabelEncoder()
            enc.fit(df[col])
            df[col] = enc.transform(df[col])
    return df


def make_gap_before_condition_met(df, gap, encounter_col_name, hour_col_name, label_col_name):
    instance_hour_df = pd.DataFrame(df[df[label_col_name] == 1].groupby(encounter_col_name)[hour_col_name].min())
    instance_hour_df = instance_hour_df.rename(columns={hour_col_name: 'instance_hour'})
    data2 = pd.merge(instance_hour_df, df, how='right', on=encounter_col_name)
    base_num_encs = data2[encounter_col_name].nunique()

    data2['instance_hour'] = data2['instance_hour'].fillna(100000)
    data2['hours_before_instance'] = data2['instance_hour'] - data2[hour_col_name]

    less_zero = data2['hours_before_instance'] <= 0
    not_in_gap = data2['hours_before_instance'] > gap
    data2 = data2[less_zero | not_in_gap]
    print("Lost " + str(
        base_num_encs - data2[encounter_col_name].nunique()) + " Encounters because condition was met at hour 0")

    data2 = data2.drop(columns=['instance_hour', 'hours_before_instance'])
    return data2


def apply_backfill(df, hours, encounter_col_name, hour_col_name, label_col_name):
    instance_hour_df = pd.DataFrame(df[df[label_col_name] == 1].groupby(encounter_col_name)[hour_col_name].min())
    instance_hour_df = instance_hour_df.rename(columns={hour_col_name: 'instance_hour'})
    data2 = pd.merge(instance_hour_df, df, how='right', on=encounter_col_name)
    base_num_encs = data2[encounter_col_name].nunique()

    data2['instance_hour'] = data2['instance_hour'].fillna(100000)
    data2['hours_before_instance'] = data2['instance_hour'] - data2[hour_col_name]
    data2 = data2[data2['hours_before_instance'] > 0]
    print("Lost " + str(
        base_num_encs - data2[encounter_col_name].nunique()) + " Encounters because condition was met at hour 0")

    data2.loc[:, label_col_name] = (data2.loc[:, 'hours_before_instance'] <= hours + 1).astype('int')
    data2 = data2.drop(columns=['instance_hour', 'hours_before_instance'])

    return data2


def read_in_chunks(path):
    chunksize = 10 ** 6
    chunks = []
    if '.csv' in path:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunks.append(optimize_dtypes(chunk, []))
        return pd.concat(chunks)
    elif '.feather' in path:
        return pd.read_feather(path)


def get_data(data_path,
             subset_columns_to_drop_duplicates,
             label_col,
             encounter_col,
             hour_col,
             hours_to_backfill,
             sample_for_prevalence,
             prevalence,
             encode_categorical,
             categorical_cols,
             make_hour_gap,
             hour_gap,
             make_weight_col,
             weight_max,
             weight_increment
             ):
    data = read_in_chunks(data_path)
    data = optimize_dtypes(data, [])
    if 'Unnamed: 0' in data.columns.values:
        data = data.drop(columns=['Unnamed: 0'])
    data = data.drop_duplicates(subset=subset_columns_to_drop_duplicates, keep='first')
    data = data.dropna()

    data = apply_backfill(
        data,
        hours_to_backfill,
        encounter_col_name=encounter_col,
        hour_col_name=hour_col,
        label_col_name=label_col)

    if sample_for_prevalence:
        data = sample_for_class_balance(data, label_col, prevalence, encounter_col)

    if encode_categorical:
        data = encode_categorical_variables(data, categorical_cols)

    if make_hour_gap:
        data = make_gap_before_condition_met(data, hour_gap, encounter_col, hour_col, label_col)

    if make_weight_col:
        data = make_weight_column(data, label_col, encounter_col, hour_col, weight_max, weight_increment)

    # Get rid of special characters in feature titles
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    return data


def make_train_test_splits_by_patients(dataset, patient_col, test_percentage=15):
    patients = list(dataset[patient_col].astype('str').unique())
    random.seed(10)
    test_patients = random.sample(patients, int((test_percentage / 100) * len(patients)))
    train_patients = np.setdiff1d(patients, test_patients)

    train_set = dataset[dataset[patient_col].isin(train_patients)]
    test_set = dataset[dataset[patient_col].isin(test_patients)]

    return train_set, test_set


def make_splits_by_patients(dataset, label_col, patient_col, test_percentage=15):
    patients = list(dataset[patient_col].astype('str').unique())
    random.seed(10)
    test_patients = random.sample(patients, int((test_percentage / 100) * len(patients)))
    train_patients = np.setdiff1d(patients, test_patients)

    Y = dataset.loc[:, [patient_col, label_col]]
    X = dataset.drop([label_col], axis=1)

    x_train = X[X[patient_col].isin(train_patients)]
    x_test = X[X[patient_col].isin(test_patients)]
    y_train = Y[Y[patient_col].isin(train_patients)]
    y_test = Y[Y[patient_col].isin(test_patients)]

    print("Test Set Percentage of Encounters: " + str(len(x_test) / dataset.shape[0]))
    x_train = x_train.drop(columns=[patient_col])
    x_test = x_test.drop(columns=[patient_col])
    y_train = y_train.drop(columns=[patient_col])
    y_test = y_test.drop(columns=[patient_col])
    return x_train, x_test, y_train, y_test


def make_weight_column_distribution(dataset, label_col, encounter_col, hour_col, max_weight, increment):
    # Make Hours before instance
    instance_hour_df = pd.DataFrame(dataset[dataset[label_col] == 1].groupby(encounter_col)[hour_col].min())
    instance_hour_df = instance_hour_df.rename(columns={hour_col: 'instance_hour'})
    data = pd.merge(instance_hour_df, dataset, how='right', on=encounter_col)
    data['hours_before_instance'] = data['instance_hour'] - data[hour_col]

    # Make weighting decline for positive encounters only
    pos_encounters = data[data[label_col] == 1][encounter_col]
    pos_data = data[data[encounter_col].isin(pos_encounters)]
    pos_data['weight'] = increment * pos_data.loc[:, ['hours_before_instance']]
    pos_weights = pos_data.loc[:, [encounter_col, hour_col, 'weight']]

    data = data.drop(columns=['instance_hour', 'hours_before_instance'])
    data = pd.merge(data, pos_weights, on=[encounter_col, hour_col], how='left')
    data = data.drop_duplicates()
    # Adjust weights
    data['weight'] = data['weight'].fillna(max_weight)  # NA will be no-instance encounters
    data['weight'] = data['weight'].clip(lower=0.0001,
                                         upper=max_weight)  # Original 1 labels will have negative weights (negative 'hours before instance')
    data['weight'] = data['weight'].replace(to_replace=0.0001, value=max_weight)
    return data


def make_weight_column(dataset, label_col, encounter_col, hour_col, pos_weight):
    dataset['weight'] = pos_weight * dataset[label_col]
    dataset['weight'] = dataset['weight'].clip(1.0)
    return dataset
