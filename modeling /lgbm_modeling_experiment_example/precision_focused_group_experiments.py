# This code allows this file to access folders in the immediate parent directory
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import lgbm_experiment_utils as lgbm_experiment_utils

# ## Gap Experiment

def run_gap_experiment(data_params, params, outcome):
    lgbm_experiment_utils.gap_experiment_lgb(data_params,
                                             gap_hour_options=[1, 2, 3, 4, 5, 6, 8],
                                             test_percentage=20,
                                             training_params=params,
                                             patient_col='PATIENT_ID',
                                             encounter_col='ENCOUNTER_ID',
                                             hour_col='hour_of_enc',
                                             label_col="Label_" + outcome,
                                             path_to_results_directory="../../results",
                                             phenotype=outcome,
                                             title="gap_experiment_1-8",
                                             description="desc",
                                             commentary="")


# ## Weighting Boundary Experiment

def run_weighting_boundary_experiment(data_params, params, outcome):
    for key in params.keys():
        params[key] = [params[key]]

    lgbm_experiment_utils.weighting_boundary_experiment_lgb(data_params,
                                                            weights=[1, 5, 10, 15, 20, 30, 50],
                                                            decline_ratios=[0.1, 0.3, 0.5],
                                                            test_percentage=20,
                                                            model_param_options=params,
                                                            patient_col='PATIENT_ID',
                                                            encounter_col='ENCOUNTER_ID',
                                                            hour_col='hour_of_enc',
                                                            label_col="Label_" + outcome,
                                                            path_to_results_directory="../../results",
                                                            phenotype=outcome,
                                                            title="weight_boundary_1_50",
                                                            description="desc",
                                                            commentary="")


# ## Prevalence Experiment

def run_prevalance_experiment(data_params, params, outcome):
    lgbm_experiment_utils.prevalence_experiment_lgb(data_params,
                                                    prevalence_options=[0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09,
                                                                        0.11, 0.15, 0.2, 0.25, 0.3, 0.35],
                                                    test_percentage=20,
                                                    model_params=params,
                                                    patient_col='PATIENT_ID',
                                                    encounter_col='ENCOUNTER_ID',
                                                    hour_col='hour_of_enc',
                                                    label_col="Label_" + outcome,
                                                    path_to_results_directory="../results",
                                                    phenotype=outcome,
                                                    title="prevalence_0.01_0.35",
                                                    description="desc",
                                                    commentary="jj")


# ## Baseline Training

def run_baseline_training(data_params, params, outcome):
    for key in params.keys():
        params[key] = [params[key]]

    lgbm_experiment_utils.test_parameter_space_lgb(data_params,
                                                   test_percentage=20,
                                                   model_param_options=params,
                                                   patient_col="PATIENT_ID",
                                                   hour_col="hour_of_enc",
                                                   encounter_col="ENCOUNTER_ID",
                                                   label_col="Label_" + outcome,
                                                   path_to_results_directory="../results",
                                                   phenotype=outcome,
                                                   title="baseline_training",
                                                   description="",
                                                   commentary="")


# ## Combined Run

def generate_precision_comparisons(data_params, params, outcome):
    run_baseline_training(data_params, params, outcome)
    run_weighting_boundary_experiment(data_params, params, outcome)
    run_gap_experiment(data_params, params, outcome)
    run_prevalance_experiment(data_params, params, outcome)
