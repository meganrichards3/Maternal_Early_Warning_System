#!/usr/bin/env python
# coding: utf-8


# ## Example Run (1 Dataset)

# params = {
#                     'path_to_encounters_file': "../data/Raw/MEWS_cohort_encounters_V2.csv",
#                     'path_to_flowsheets_file' : "../data/mews_enc_flowsheets_cleaned.csv",
#                     'path_to_icd10_file' : '../data/mews_pipeline_mrn_icd10.csv',
#                     'path_to_problem_list_file' : "../data/mews_pipeline_patients_problem_list.csv",
#                     'path_to_icd_grouping_file' : "../data/icd_ccs_full.csv",
#                     'path_to_analyte_file' : '../data/mews_enc_analytes_all.csv',
#                     'path_to_analyte_grouping_file' : '../data/analytes_enterprise_groupers_11_29_2020.csv',
#                     'path_to_orders_file' : '../data/clean_orders_fp.csv',
#                     'path_to_prenatal_enc_file' : "../data/mews_pipeline_mrn_prenatal_encs.csv",
#                     'save' :True,
#                     'hours_to_carry': 1,
#                     'months_to_lookback' :12,
#                      'include_race': True,
#                     'include_icd' :True,
#                     'include_problem_list':False,
#                     'icd_percent_threshold': 0.1,
#                     'analyte_percent_threshold': 0.1,
#                     'drop_non_grouped_analytes': True,
#                     'remove_sparse_columns' : True,
#                     'path_to_outcomes_file': "../processed_data/outcomes.csv",
#                      'save_full_as' : "../processed_data/base_datasets/"
# }
# a =make_full_dataset(**params)


# ## Example Parameter Search

# baseline_params = {
#                     'path_to_encounters_file': "../../mews/Data/Raw/MEWS_cohort_encounters_V2.csv",
#                     'path_to_flowsheets_file' : "../data/mews_enc_flowsheets_cleaned.csv",
#                     'path_to_icd10_file' : '../data/mews_pipeline_mrn_icd10.csv',
#                     'path_to_problem_list_file' : "../data/mews_pipeline_patients_problem_list.csv",
#                     'path_to_icd_grouping_file' : "../data/icd_ccs_full.csv",
#                     'path_to_analyte_file' : '../data/mews_enc_analytes_all.csv',
#                     'path_to_analyte_grouping_file' : '../data/analytes_enterprise_groupers_11_29_2020.csv',
#                     'path_to_orders_file' : '../data/clean_orders_fp.csv',
#                     'path_to_prenatal_enc_file' : "../data/mews_pipeline_mrn_prenatal_encs.csv",
#                     'save' :True,
# }
# options = {
#         'hours_to_carry': [1,3,5,7,15,50],
#         'months_to_lookback' :[3,6,9,12,18],
#          'include_race': [True, False],
#         'include_icd' :[True, False],
#         'include_problem_list':[True, False],
#         'icd_percent_threshold': [0.001, 0.005, 0.01, 0.05],
#         'analyte_percent_threshold': [0.001, 0.005, 0.01, 0.05],
#         'drop_non_grouped_analytes': [True, False],
#         'remove_sparse_columns' : [True, False],
#         'path_to_outcomes_file': ["../processed_data/outcomes.csv"],
#          'save_full_as' :["../processed_data/base_datasets/"]
# }


import pandas as pd
import numpy as np
from typing import List
from scipy import stats
import csv
import datetime
import itertools


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


def optimize(df: pd.DataFrame, datetime_features: List[str] = []):
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))


def write_documentation_row(experiments_file_path, row_to_add):
    with open(experiments_file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row_to_add.values())


def get_patients(path_to_encounters_file, months_to_lookback, include_race):
    patients = pd.read_csv(path_to_encounters_file)
    patients = patients.loc[:,
               ['ENCOUNTER_ID', 'PATIENT_ID', "AGE_AT_ADM", "RACE", "HOSPITAL_ADMITTED_AT", "INPATIENT_ADMITTED_AT"]]
    patients['HOSPITAL_ADMITTED_AT'] = pd.to_datetime(patients['HOSPITAL_ADMITTED_AT'])
    patients['INPATIENT_ADMITTED_AT'] = pd.to_datetime(patients['INPATIENT_ADMITTED_AT'])

    # For 3 patients who don't have inpatient_admitted_time
    patients['INPATIENT_ADMITTED_AT'].fillna(patients['HOSPITAL_ADMITTED_AT'], inplace=True)

    # Change types
    patients['AGE_AT_ADM'] = patients['AGE_AT_ADM'].astype('int64')
    patients['ENCOUNTER_ID'] = patients['ENCOUNTER_ID'].astype('int64')
    patients = optimize(patients, ['HOSPITAL_ADMITTED_AT', 'INPATIENT_ADMITTED_AT'])
    patients['RACE'] = patients['RACE'].astype('object')

    # Binarize Race column
    if include_race:
        patients['RACE-BLACK'] = patients['RACE'].str.contains('Black').fillna(0).astype('int8')
        patients['RACE-WHITE'] = patients['RACE'].str.contains('White').fillna(0).astype('int8')
        patients['RACE-OTHER'] = patients['RACE'].str.contains('Other').fillna(0).astype('int8')
        patients['RACE-ASIAN'] = patients['RACE'].str.contains('Asian').fillna(0).astype('int8')
        patients['RACE-PACIFIC'] = patients['RACE'].str.contains('Pacific').fillna(0).astype('int8')
        patients['RACE-NATIVE'] = patients['RACE'].str.contains('Native').fillna(0).astype('int8')
        patients['RACE-MULTIPLE'] = patients['RACE'].str.contains('2').fillna(0).astype('int8')
    patients.drop(columns='RACE', inplace=True)

    # Exclude Patients who don't have the lookback period
    patients = patients[
        patients['INPATIENT_ADMITTED_AT'] > np.datetime64("2015-01-01") + pd.Timedelta(days=30 * months_to_lookback)]

    return patients


def get_flowsheets(path_to_flowsheets_file, encounter_cohort):
    flowsheets = pd.read_csv(path_to_flowsheets_file)
    flowsheets = flowsheets.loc[:,
                 ['Grouper', 'Name', 'encounter_id', 'normalized_unit', 'normalized_value', 'recorded_at']]
    flowsheets['recorded_at'] = pd.to_datetime(flowsheets['recorded_at'])
    flowsheets.dropna(subset=['encounter_id'], inplace=True)
    flowsheets['encounter_id'] = flowsheets['encounter_id'].astype('int64')

    flowsheets.rename(columns={'recorded_at': 'Time', 'normalized_value': 'Value'}, inplace=True)
    flowsheets = flowsheets[['Grouper', 'encounter_id', 'Time', 'Value']]
    flowsheets['Grouper'] = flowsheets['Grouper'] + '_flowsheet'

    flowsheets = optimize(flowsheets, ['recorded_at'])
    flowsheets = flowsheets[flowsheets['encounter_id'].isin(encounter_cohort)]
    flowsheets.dropna(inplace=True)
    return flowsheets


def get_comorbidities(path_to_icd10_file, path_to_problem_list_file, path_to_icd_grouping_file, patients, include_icd,
                      include_problem_list, threshold, months):
    # Get comorbidities
    comorb = pd.DataFrame(columns=['PATIENT_ID', 'ENCOUNTER_EFFECTIVE_AT', 'CODE'])
    if include_icd:
        icd = pd.read_csv(path_to_icd10_file)
        icd = icd.loc[:, ['PATIENT_ID', 'ENCOUNTER_EFFECTIVE_AT', 'CODE']]
        comorb = pd.concat([comorb, icd], ignore_index=True)
    if include_problem_list:
        prob = pd.read_csv(path_to_problem_list_file)
        prob.rename({"CURRENT_ICD10_LIST": "CODE"}, axis=1, inplace=True)
        prob = prob.loc[:, ['PATIENT_ID', 'ENCOUNTER_EFFECTIVE_AT', 'CODE']]
        comorb = pd.concat([comorb, prob], ignore_index=True)

    # Group Codes
    comorb['ENCOUNTER_EFFECTIVE_AT'] = pd.to_datetime(comorb['ENCOUNTER_EFFECTIVE_AT'])
    icd_grouping = pd.read_csv(path_to_icd_grouping_file)
    icd_mapping_dict = pd.Series(icd_grouping.loc[:, 'category'].values, index=icd_grouping.loc[:, 'code']).to_dict()
    comorb.loc[:, 'CODE'] = comorb.loc[:, 'CODE'].str.replace(r'[^\w\s]+', '', regex=True)
    comorb.loc[:, 'Comorbidities'] = comorb.loc[:, 'CODE'].map(icd_mapping_dict)

    # Combine with patients
    comorb = comorb[comorb['PATIENT_ID'].isin(patients['PATIENT_ID'].unique())]
    comorb = pd.merge(patients, comorb, on=['PATIENT_ID'], how='left')

    # Filter for prevalence and time
    comorb = comorb.groupby("Comorbidities").filter(
        lambda x: x['PATIENT_ID'].nunique() > threshold * comorb['PATIENT_ID'].nunique())
    comorb['THRESHOLD_DATE'] = comorb['INPATIENT_ADMITTED_AT'] - pd.Timedelta(days=30 * months)
    comorb = comorb[comorb['ENCOUNTER_EFFECTIVE_AT'] > comorb['THRESHOLD_DATE']]
    comorb = comorb[comorb['ENCOUNTER_EFFECTIVE_AT'] < (comorb['INPATIENT_ADMITTED_AT'] - pd.Timedelta(days=1))]

    # Clean and return
    comorb = optimize(comorb, [])
    comorb = comorb.drop(columns='CODE')
    print("Comorbidity Groupers = " + str(comorb['Comorbidities'].nunique()))
    if include_icd:
        del icd
    if include_problem_list:
        del prob
    del icd_grouping
    return comorb


def map_analyte_groupers(path_to_analyte_file, path_to_analyte_grouping_file, drop_non_grouped_analytes):
    all_analytes = pd.read_csv(path_to_analyte_file)
    mapping_df = pd.read_csv(path_to_analyte_grouping_file)
    mapping_df = mapping_df.loc[:, ['component_id', 'grouper']].drop_duplicates()

    conversion_dict = pd.Series(mapping_df['grouper'].values, index=mapping_df['component_id']).to_dict()
    all_analytes.loc[:, 'GROUPER'] = all_analytes.loc[:, 'COMPONENT_ID'].map(conversion_dict)
    print("Not Grouped: " + str(all_analytes['GROUPER'].isna().sum()))
    print("Total : " + str(all_analytes.shape[0]))
    print("Percentage of Analytes Grouped: " + str(1 - all_analytes['GROUPER'].isna().sum() / all_analytes.shape[0]))

    if drop_non_grouped_analytes:
        all_analytes = all_analytes.dropna(subset=['GROUPER'])
    else:
        all_analytes['GROUPER'].fillna(all_analytes['COMMON_NAME'], inplace=True)
    return all_analytes


def get_analytes(path_to_analyte_file, path_to_analyte_grouping_file, drop_non_grouped_analytes, encounter_cohort,
                 threshold):
    analytes = map_analyte_groupers(path_to_analyte_file, path_to_analyte_grouping_file, drop_non_grouped_analytes)

    analytes = analytes[analytes['ENCOUNTER_ID'].isin(encounter_cohort)]
    analytes = analytes.groupby("GROUPER").filter(
        lambda x: x['ENCOUNTER_ID'].nunique() > threshold * analytes['ENCOUNTER_ID'].nunique())

    # Split analytes into results and orders
    analyte_results = analytes.copy()
    analyte_orders = analytes.rename(
        columns={'COLLECTED_AT': 'Time', 'GROUPER': 'Grouper', 'ENCOUNTER_ID': 'encounter_id'})
    analyte_orders.loc[:, 'Value'] = 1

    analyte_results.rename(
        columns={'RESULTED_AT': 'Time', 'GROUPER': 'Grouper', 'ENCOUNTER_ID': 'encounter_id', 'RAW_VALUE': 'Value'},
        inplace=True)
    analyte_orders = analyte_orders.loc[:, ['Grouper', 'encounter_id', 'Time', 'Value']]

    analyte_results = analyte_results.loc[:, ['Grouper', 'encounter_id', 'Time', 'Value']]
    analyte_results.dropna(how='any', inplace=True)
    analyte_orders.dropna(how='any', inplace=True)

    analyte_orders['Grouper'] = analyte_orders['Grouper'] + '_analyte_collected'
    analyte_results.loc[:, 'Grouper'] = analyte_results.loc[:, 'Grouper'] + '_analyte_results'

    analytes_final = pd.concat([analyte_results, analyte_orders])
    analytes_final['encounter_id'] = analytes_final['encounter_id'].astype('int64')
    analytes_final_optimized = optimize(analytes_final, ['Time'])
    del analytes, analyte_results, analyte_orders
    return analytes_final_optimized


def get_orders(path_to_orders_file, encounter_cohort):
    orders = pd.read_csv(path_to_orders_file)
    orders.rename(columns={'ORDERED_AT': 'Time', 'GROUPER': 'Grouper', 'ENCOUNTER_ID': 'encounter_id'}, inplace=True)
    orders.loc[:, 'Value'] = 1
    orders = orders[['Grouper', 'encounter_id', 'Time', 'Value']]
    orders['Grouper'] = orders['Grouper'] + '_order'
    orders['encounter_id'] = orders['encounter_id'].astype('int64')
    orders = orders[orders['encounter_id'].isin(encounter_cohort)]
    orders = optimize(orders, ['Time'])
    return orders


def make_prenatal_patient_set(path_to_prenatal_enc_file, patients_raw):
    prenatal = pd.read_csv(path_to_prenatal_enc_file).loc[:,
               ['PATIENT_ID', 'ENCOUNTER_EFFECTIVE_AT', 'REASON_LABEL', 'ENC_TYPE_LABEL', 'PRIMARY_DIAGNOSIS']]
    prenatal['ENCOUNTER_EFFECTIVE_AT'] = pd.to_datetime(prenatal['ENCOUNTER_EFFECTIVE_AT'])
    prenatal.rename(columns={'ENCOUNTER_EFFECTIVE_AT': 'PRENATAL_TIME'}, inplace=True)
    patients_prenatal = pd.merge(patients_raw, prenatal, how='left', left_on=['PATIENT_ID'], right_on=['PATIENT_ID'])

    # Split for those with/without prenatal info
    encounters_without_prenatal = patients_prenatal[patients_prenatal['PRENATAL_TIME'].isna()]
    encounters_with_prenatal = patients_prenatal[~patients_prenatal['PRENATAL_TIME'].isna()]

    ##PRENATAL

    # Filter prenatal cases
    encounters_with_prenatal = encounters_with_prenatal[
        encounters_with_prenatal['INPATIENT_ADMITTED_AT'] > encounters_with_prenatal['PRENATAL_TIME']]
    encounters_with_prenatal = encounters_with_prenatal[
        encounters_with_prenatal['PRENATAL_TIME'] > encounters_with_prenatal['INPATIENT_ADMITTED_AT'] - pd.Timedelta(
            days=30 * 10)]

    # Make features
    num_encounters = encounters_with_prenatal.groupby('ENCOUNTER_ID').apply(
        lambda x: (x['PRENATAL_TIME'].nunique())).reset_index()
    num_diagnoses = encounters_with_prenatal.groupby('ENCOUNTER_ID').apply(
        lambda x: (x['PRIMARY_DIAGNOSIS'].nunique())).reset_index()
    most_frequent_diagnosis = encounters_with_prenatal.groupby('ENCOUNTER_ID')['PRIMARY_DIAGNOSIS'].apply(
        lambda x: stats.mode(x)[0][0]).reset_index()
    num_encounters.rename(columns={0: 'NUMBER_PRENATAL_VISITS'}, inplace=True)
    num_diagnoses.rename(columns={0: 'NUMBER_PRENATAL_DIAGNOSES'}, inplace=True)
    most_frequent_diagnoses = most_frequent_diagnosis.rename(
        columns={'PRIMARY_DIAGNOSIS': 'MOST_FREQUENT_PRENATAL_DIAGNOSIS'})

    new_features = pd.merge(num_encounters, num_diagnoses, how='outer', on='ENCOUNTER_ID')
    new_features = pd.merge(new_features, most_frequent_diagnoses, how='outer', on='ENCOUNTER_ID')

    encounters_with_prenatal.drop(columns=['PRIMARY_DIAGNOSIS', 'REASON_LABEL', 'ENC_TYPE_LABEL',
                                           'PRENATAL_TIME'], inplace=True)
    encounters_with_prenatal.drop_duplicates(inplace=True)
    prenatal_encounters_with_features = pd.merge(new_features, encounters_with_prenatal, on='ENCOUNTER_ID', how='right')

    # Split for those with/without prenatal info
    encounters_without_prenatal = patients_prenatal[
        ~patients_prenatal['ENCOUNTER_ID'].isin(prenatal_encounters_with_features['ENCOUNTER_ID'])]

    encounters_without_prenatal.drop(columns=['PRIMARY_DIAGNOSIS', 'REASON_LABEL', 'ENC_TYPE_LABEL', 'PRENATAL_TIME'],
                                     inplace=True)
    encounters_without_prenatal.drop_duplicates(inplace=True)

    ##NOTPRENATAL

    # Make Features
    encounters_without_prenatal.loc[:, 'NUMBER_PRENATAL_VISITS'] = 0
    encounters_without_prenatal.loc[:, 'NUMBER_PRENATAL_DIAGNOSES'] = 0
    encounters_without_prenatal.loc[:, 'MOST_FREQUENT_PRENATAL_DIAGNOSIS'] = " "

    # Recombine prenatal and non-prenatal
    combined_encounters = pd.concat([prenatal_encounters_with_features, encounters_without_prenatal])
    combined_encounters.drop_duplicates(inplace=True)

    combined_encounters = combined_encounters.drop_duplicates(subset='ENCOUNTER_ID', keep='first')
    del patients_prenatal, num_encounters, num_diagnoses, most_frequent_diagnosis, new_features, encounters_with_prenatal, encounters_without_prenatal
    return combined_encounters


def add_time_processing(dataset):
    dataset.loc[:, 'wait_before_admission'] = (dataset.loc[:, 'HOSPITAL_ADMITTED_AT'] - dataset.loc[:,
                                                                                        'INPATIENT_ADMITTED_AT']).dt.total_seconds() / 60 / 60
    dataset.loc[:, 'hour_of_enc'] = (dataset['Time'] - dataset['INPATIENT_ADMITTED_AT']).dt.total_seconds() / 60 / 60
    dataset.dropna(how='any', subset=['encounter_id', 'Value', 'Grouper', 'hour_of_enc', 'INPATIENT_ADMITTED_AT'],
                   inplace=True)  # .copy()
    dataset.loc[:, 'hour_of_enc'] = dataset.loc[:, 'hour_of_enc'].apply(np.ceil)
    dataset.loc[:, 'hour_of_enc'] = dataset.loc[:, 'hour_of_enc'].clip(lower=0)

    return dataset


def process_time_data(combined_time_data, patients, hours_to_carry):
    patient_time_data = pd.merge(combined_time_data, patients, how='right', left_on='encounter_id',
                                 right_on='ENCOUNTER_ID')
    patient_time_data = add_time_processing(patient_time_data)

    data_by_patient_hour = patient_time_data.pivot_table(
        index=['encounter_id', 'hour_of_enc'],
        columns='Grouper',
        values=['Value'])
    max_hour_df = pd.DataFrame(data_by_patient_hour.reset_index().groupby('encounter_id')['hour_of_enc'].max())
    max_hour_df = max_hour_df.reindex(max_hour_df.index.repeat(max_hour_df['hour_of_enc']))
    max_hour_df['hour_of_enc'] = max_hour_df.groupby('encounter_id').cumcount() + 1
    max_hour_df = max_hour_df.reset_index()
    data_by_patient_hour.columns = data_by_patient_hour.columns.get_level_values(1)
    all_hours = pd.merge(max_hour_df, data_by_patient_hour, how='left', left_on=['encounter_id', 'hour_of_enc'],
                         right_on=['encounter_id', 'hour_of_enc'])
    all_hours = all_hours.set_index('encounter_id')
    all_hours = all_hours.groupby('encounter_id').ffill(hours_to_carry)

    # forward fill height/weight/bmi fully
    cols = ['weight_flowsheet', 'temperature_flowsheet', 'height_flowsheet', 'bmi_flowsheet']
    all_hours.loc[:, cols] = all_hours.loc[:, cols].ffill()

    all_hours.fillna(0, inplace=True)

    all_hours.reset_index(inplace=True)
    del data_by_patient_hour, max_hour_df, patient_time_data
    return all_hours


def fill_bmi(dataset):
    columns = ['weight_flowsheet', 'temperature_flowsheet', 'height_flowsheet', 'bmi_flowsheet']
    return dataset.loc[:, columns].ffill()


# In[4]:


def get_time_data(path_to_encounters_file,
                  path_to_flowsheets_file,
                  path_to_analyte_file,
                  path_to_analyte_grouping_file,
                  path_to_orders_file,
                  months_lookback,
                  include_race,
                  hours_to_carry,
                  analyte_threshold,
                  drop_non_grouped_analytes):
    patients = get_patients(path_to_encounters_file, months_lookback, include_race)
    flowsheets = get_flowsheets(path_to_flowsheets_file, patients['ENCOUNTER_ID'].unique())
    orders = get_orders(path_to_orders_file, patients['ENCOUNTER_ID'].unique())
    analytes = get_analytes(path_to_analyte_file, path_to_analyte_grouping_file, drop_non_grouped_analytes,
                            patients['ENCOUNTER_ID'].unique(), analyte_threshold)

    combined = pd.concat([flowsheets, orders, analytes])
    combined['Time'] = pd.to_datetime(combined['Time'])

    time_data = process_time_data(combined, patients, hours_to_carry)

    # Make data small!
    time_data.set_index(['encounter_id', 'hour_of_enc'], inplace=True)
    time_data = time_data.astype('int64')
    time_data.reset_index(inplace=True)
    time_data['encounter_id'] = time_data['encounter_id'].astype('int64')
    time_data['hour_of_enc'] = time_data['hour_of_enc'].astype('int64')

    time_data = optimize(time_data, ['Time'])

    del flowsheets, orders, analytes, patients, combined
    return time_data


def get_time_invariant_data(path_to_encounters_file, path_to_icd10_file, path_to_problem_list_file,
                            path_to_icd_grouping_file, path_to_prenatal_enc_file,
                            months, include_race, include_icd, include_problem_list, icd_percent_threshold):
    # Get Data
    patients_raw = get_patients(path_to_encounters_file, months, include_race)
    icd10s = get_comorbidities(path_to_icd10_file,
                               path_to_problem_list_file,
                               path_to_icd_grouping_file,
                               patients_raw,
                               include_icd=include_icd,
                               include_problem_list=include_problem_list,
                               threshold=icd_percent_threshold,
                               months=months
                               )
    patients_prenatal = make_prenatal_patient_set(path_to_prenatal_enc_file, patients_raw)

    # Pivot to binarize comorbidities data
    eff = icd10s.loc[:, ['PATIENT_ID', 'ENCOUNTER_EFFECTIVE_AT']].drop_duplicates()
    icd10s['Ones'] = 1
    icd10s = icd10s.drop(columns=['ENCOUNTER_EFFECTIVE_AT']).pivot_table(index=['PATIENT_ID'], columns='Comorbidities',
                                                                         values=['Ones'])
    icd10s = icd10s.fillna(0)
    icd10s.drop_duplicates(inplace=True)
    icd10s.columns = icd10s.columns.get_level_values(1)
    icd10s.columns = list(icd10s.columns)
    eff.columns = list(eff.columns)
    icd10s = pd.merge(icd10s, eff, on='PATIENT_ID')

    # Make Time Invariant Data
    time_invariant_data = pd.merge(patients_prenatal, icd10s, how='left', left_on=['PATIENT_ID'],
                                   right_on=['PATIENT_ID'])
    time_invariant_data['THRESHOLD_DATE'] = time_invariant_data['INPATIENT_ADMITTED_AT'] - pd.Timedelta(
        days=30 * months)
    time_invariant_data = time_invariant_data[
        time_invariant_data['ENCOUNTER_EFFECTIVE_AT'] > time_invariant_data['THRESHOLD_DATE']]
    time_invariant_data = time_invariant_data[time_invariant_data['ENCOUNTER_EFFECTIVE_AT'] < (
                time_invariant_data['INPATIENT_ADMITTED_AT'] - pd.Timedelta(days=1))]
    l = list(patients_prenatal.columns.values)
    time_invariant_data = pd.merge(patients_prenatal, time_invariant_data, how='left', left_on=l, right_on=l)
    time_invariant_data.drop_duplicates(inplace=True)
    time_invariant_data = time_invariant_data.drop(
        columns=['ENCOUNTER_EFFECTIVE_AT', 'THRESHOLD_DATE']).drop_duplicates()

    time_invariant_data = time_invariant_data.fillna(0)

    all_columns = time_invariant_data.columns.values
    non_history = ['ENCOUNTER_ID', 'PATIENT_ID', 'hour_of_enc', 'AGE_AT_ADM', 'RACE',
                   'MOST_FREQUENT_PRENATAL_DIAGNOSIS', 'NUMBER_PRENATAL_DIAGNOSES', 'NUMBER_PRENATAL_VISITS',
                   'HOSPITAL_ADMITTED_AT', 'INPATIENT_ADMITTED_AT']
    non_history = non_history + [i for i in all_columns if 'order' in i]
    non_history = non_history + [i for i in all_columns if 'flowsheet' in i]
    non_history = non_history + [i for i in all_columns if 'result' in i]
    history_cols = [ele for ele in all_columns if ele not in non_history]

    for col in history_cols:
        time_invariant_data.loc[:, col] = time_invariant_data.loc[:, col].astype('int32')

    time_invariant_data = optimize(time_invariant_data, ['INPATIENT_ADMITTED_AT', 'HOSPITAL_ADMITTED_AT'])
    del patients_prenatal, eff, icd10s
    return time_invariant_data


def combine_all_data_and_clean(time_invariant_data, time_variant_data):
    # Combine
    time_encounters = time_variant_data['encounter_id']
    non_time_encounters = time_invariant_data['ENCOUNTER_ID']

    print("Time Encounters = " + str(time_encounters.nunique()))
    print("Non Time Encounters = " + str(non_time_encounters.nunique()))

    input_dataset = pd.merge(time_invariant_data, time_variant_data, how='right', left_on=['ENCOUNTER_ID'],
                             right_on=['encounter_id'])

    input_dataset.drop(columns=['HOSPITAL_ADMITTED_AT', 'INPATIENT_ADMITTED_AT', 'encounter_id'], inplace=True)

    # Remove duplicates, drop NA
    input_dataset.drop_duplicates(inplace=True)
    input_dataset.dropna(subset=['ENCOUNTER_ID', 'PATIENT_ID', 'hour_of_enc'], how='any', inplace=True)

    # Make history int8
    all_columns = input_dataset.columns.values
    non_history = ['ENCOUNTER_ID', 'PATIENT_ID', 'hour_of_enc', 'AGE_AT_ADM', 'RACE',
                   'MOST_FREQUENT_PRENATAL_DIAGNOSIS', 'NUMBER_PRENATAL_DIAGNOSES', 'NUMBER_PRENATAL_VISITS']

    non_history = non_history + [i for i in all_columns if 'order' in i and 'isorder' not in i]
    non_history = non_history + [i for i in all_columns if 'flowsheet' in i]
    non_history = non_history + [i for i in all_columns if 'result' in i]

    history_cols = [ele for ele in all_columns if ele not in non_history]

    for col in history_cols:
        input_dataset[col] = input_dataset[col].astype('int8')

    del time_encounters, non_time_encounters

    return input_dataset


def clean_sparse_columns(inputs, threshold):
    data = inputs.drop_duplicates(subset='PATIENT_ID', keep='first')
    columns_to_keep = []
    for column in inputs.columns.values:
        nonzero = np.count_nonzero(data[column])
        if nonzero > threshold:
            columns_to_keep.append(column)
    inputs = inputs.loc[:, columns_to_keep]
    del data
    return inputs


def get_input_data(
        path_to_encounters_file,
        path_to_flowsheets_file,
        path_to_icd10_file,
        path_to_problem_list_file,
        path_to_icd_grouping_file,
        path_to_analyte_file,
        path_to_analyte_grouping_file,
        path_to_orders_file,
        path_to_prenatal_enc_file,
        hours_to_carry,
        months_to_lookback,
        include_race,
        include_icd,
        include_problem_list,
        icd_percent_threshold,
        analyte_percent_threshold,
        drop_non_grouped_analytes,
        save,
        save_as,
        remove_sparse_columns):
    # Time Data
    time_data = get_time_data(path_to_encounters_file,
                              path_to_flowsheets_file,
                              path_to_analyte_file,
                              path_to_analyte_grouping_file,
                              path_to_orders_file,
                              months_to_lookback,
                              include_race,
                              hours_to_carry,
                              analyte_percent_threshold,
                              drop_non_grouped_analytes)

    # Get Time Invariant Data
    time_invariant_data = get_time_invariant_data(
        path_to_encounters_file,
        path_to_icd10_file,
        path_to_problem_list_file,
        path_to_icd_grouping_file,
        path_to_prenatal_enc_file,
        months_to_lookback,
        include_race,
        include_icd=include_icd,
        include_problem_list=include_problem_list,
        icd_percent_threshold=icd_percent_threshold)

    # Combine data + drop N/A, remove admit time
    input_dataset = combine_all_data_and_clean(time_invariant_data, time_data)

    # Remove sparse columns
    if remove_sparse_columns:
        input_dataset = clean_sparse_columns(input_dataset, 1000)

    # save
    if save:
        input_dataset.to_csv(save_as, index=False)

    input_dataset = optimize(input_dataset, [])
    del time_data, time_invariant_data
    return input_dataset


def make_full_dataset(
        path_to_encounters_file,
        path_to_flowsheets_file,
        path_to_icd10_file,
        path_to_problem_list_file,
        path_to_icd_grouping_file,
        path_to_analyte_file,
        path_to_analyte_grouping_file,
        path_to_orders_file,
        path_to_prenatal_enc_file,
        hours_to_carry,
        months_to_lookback,
        include_race,
        include_icd,
        include_problem_list,
        icd_percent_threshold,
        analyte_percent_threshold,
        drop_non_grouped_analytes,
        save,
        save_full_as,
        remove_sparse_columns,
        path_to_outcomes_file,
):
    inputs = get_input_data(
        path_to_encounters_file,
        path_to_flowsheets_file,
        path_to_icd10_file,
        path_to_problem_list_file,
        path_to_icd_grouping_file,
        path_to_analyte_file,
        path_to_analyte_grouping_file,
        path_to_orders_file,
        path_to_prenatal_enc_file,
        hours_to_carry,
        months_to_lookback,
        include_race,
        include_icd,
        include_problem_list,
        icd_percent_threshold,
        analyte_percent_threshold,
        drop_non_grouped_analytes,
        save=False,
        save_as="",
        remove_sparse_columns=remove_sparse_columns)

    outcomes = pd.read_csv(path_to_outcomes_file)

    outcomes.set_index(['ENCOUNTER_ID', 'hour_of_enc'], inplace=True)
    outcomes = outcomes.add_prefix("Label_")
    outcomes.reset_index(inplace=True)
    outcomes = optimize(outcomes, [])

    outcomes.reset_index(inplace=True)
    inputs.reset_index(inplace=True)

    inputs.drop(columns=['index'], inplace=True)
    outcomes.drop(columns=['index'], inplace=True)

    combined = pd.merge(inputs, outcomes, how='left', on=['ENCOUNTER_ID', 'hour_of_enc'])

    label_cols = [col for col in combined.columns.values if col.startswith('Label')]
    combined.loc[:, label_cols] = combined.loc[:, label_cols].fillna(0)

    combined = optimize(combined, [])
    combined.drop_duplicates(inplace=True)
    combined.reset_index(inplace=True)
    combined['MOST_FREQUENT_PRENATAL_DIAGNOSIS'] = combined['MOST_FREQUENT_PRENATAL_DIAGNOSIS'].astype('str')
    if save:
        # Save Base Dataset
        combined.to_feather(save_full_as)

        # Save Phenotype Datasets
        split_data_into_phenotypes_and_save(
            combined,
            path_to_processed_data_folder=save_full_as.split('base')[0],
            file_name=save_full_as.split('datasets/')[1],
            outcome_labels=['Label_ahf', 'Label_ards', 'Label_arf', 'Label_hemorrhage',
                            'Label_dic', 'Label_eclampsia', 'Label_embolism',
                            'Label_sepsis']
        )
    del outcomes
    del inputs
    del combined

    return save_full_as.split('datasets/')[1]


def split_data_into_phenotypes_and_save(dataset, path_to_processed_data_folder, file_name, outcome_labels):
    for outcome in outcome_labels:
        subset_outcomes = outcome_labels.copy()
        subset_outcomes.remove(outcome)
        subset = dataset.copy()
        subset.drop(subset_outcomes, axis=1)
        subset.to_feather(
            path_to_processed_data_folder + "/" + outcome.removeprefix("Label_") + "_datasets/" + outcome.removeprefix(
                "Label_") + "_" + file_name)


def dataset_param_search(dataset_prefix, constant_params, changing_params):
    i = 0
    args = changing_params.values()
    keys = list(changing_params.keys())
    for combination in itertools.product(*args):
        c = {keys[i]: combination[i] for i in range(len(keys))}
        params = {**constant_params, **c}
        if params['include_icd'] != params['include_problem_list']:
            params['save_full_as'] = params['save_full_as'] + dataset_prefix + str(i) + ".feather"
            i = i + 1

            try:
                make_full_dataset(**params)

                doc = {
                    "Unnamed": 0,
                    "Dataset Name": [params['save_full_as']],
                    "File/Function Used": ["generate_mews_datasets.param_search"],
                    "Date": [datetime.datetime.now().strftime("%m-%b-%Y")],
                    "Arguments": params}

                write_documentation_row("../processed_data/dataset_documentation.csv", doc)
                print('')
                print("Made dataset " + str(i))

            except Exception as e:
                print("Could not create dataset with parameters : ")
                print(params)
                print("Because of Exception :")
                print(e)