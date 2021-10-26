import generate_mews_dataset_utils as utils

constant_params = {
    'path_to_encounters_file': "../../mews/Data/Raw/MEWS_cohort_encounters_V2.csv",
    'path_to_flowsheets_file': "../data/mews_enc_flowsheets_cleaned.csv",
    'path_to_icd10_file': '../data/mews_pipeline_mrn_icd10.csv',
    'path_to_problem_list_file': "../data/mews_pipeline_patients_problem_list.csv",
    'path_to_icd_grouping_file': "../data/icd_ccs_full.csv",
    'path_to_analyte_file': '../data/mews_enc_analytes_all.csv',
    'path_to_analyte_grouping_file': '../data/analytes_enterprise_groupers_11_29_2020.csv',
    'path_to_orders_file': '../data/clean_orders_fp.csv',
    'path_to_prenatal_enc_file': "../data/mews_pipeline_mrn_prenatal_encs.csv",
    'save': True,
}
changing_params = {
    'hours_to_carry': [3, 5, 10, 20],
    'months_to_lookback': [6, 12, 18],
    'include_race': [True, False],
    'include_icd': [True, False],
    'include_problem_list': [True, False],
    'icd_percent_threshold': [0.001, 0.005, 0.01],
    'analyte_percent_threshold': [0.001, 0.005, 0.01],
    'drop_non_grouped_analytes': [True],
    'remove_sparse_columns': [False],
    'path_to_outcomes_file': ["../processed_data/outcomes.csv"],
    'save_full_as': ["../processed_data/base_datasets/"]
}

utils.dataset_param_search('example', constant_params, changing_params)
