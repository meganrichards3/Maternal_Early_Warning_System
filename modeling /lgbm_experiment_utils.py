import data_utils
import csv
import itertools
import os
from datetime import datetime
import pandas as pd
from sklearn import metrics
import lightgbm as lgb
import re


def write_experiments(experiments_file_path, row_to_add):
    with open(experiments_file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row_to_add.values())


def train(train_set,
          test_set,
          label_col,
          encounter_col,
          patient_col,
          hour_col,
          model_params,
          has_weight_col,
          path_to_results_directory,
          save,
          data_params_for_logging,
          title,
          phenotype,
          values_tested,
          description="",
          commentary=""):
    exp_results_folder_name = str(datetime.now().strftime("%m-%d-%Y %H:%M")) + "-" + title
    exp_results_folder_path = path_to_results_directory + "/" + phenotype + "/" + exp_results_folder_name

    # Split datasets into input and labels
    y_train = train_set[label_col]
    x_train = train_set.drop(columns=label_col)

    y_test = test_set[label_col]
    x_test = test_set.drop(columns=label_col)

    # Remove weight column from inputs
    if has_weight_col:
        train_weight = x_train['weight']
        x_train = x_train.drop(columns='weight')
        lgb_train = lgb.Dataset(x_train, y_train, weight=train_weight)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    else:
        # Create Datasets
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    # Make params
    baseline_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'verbose': 0,
        'task': 'train',
    }

    training_params = {**baseline_params, **model_params}
    model = lgb.train(training_params, lgb_train, valid_sets=lgb_eval)

    # Evaluate Results and Save
    y_pred = model.predict(x_test, num_iteration=model.best_iteration)
    auroc = metrics.roc_auc_score(y_test, y_pred)
    avg_prec = metrics.average_precision_score(y_test, y_pred)

    combs = pd.DataFrame()

    # Make row to save
    exp_run = {
        'Title': title,
        'Function Used': 'train_lgb',
        'Date': str(datetime.now().strftime("%m/%d/%Y %H:%M:%S")),
        'Combined Training Params': str(training_params)
    }
    exp_run = {**exp_run, **data_params_for_logging, **training_params, 'auroc': auroc, 'avg_prec': avg_prec}

    combs = combs.append(exp_run, ignore_index=True)

    predictions = pd.DataFrame({
        'encounter_id': list(x_test[encounter_col]),
        'patient_id': list(x_test[patient_col]),
        'hour': list(x_test[hour_col]),
        'y_pred': list(y_pred),
        'y_test': list(y_test),
    })

    best_auc = {
        'metric': auroc,
        'predictions': predictions,
        'model': model,
    }

    best_prec = {
        'metric': avg_prec,
        'predictions': predictions,
        'model': model,
    }

    if save:
        os.mkdir(exp_results_folder_path)
        # Save Experiment Results
        row_to_add = {

            'Title': title,
            'Description': description,
            'Function Used': 'train_lgb',
            'Values Tested': values_tested,
            'Data Params': str(data_params_for_logging),
            'Best AUC Params': str(training_params),
            'Best Prec Params': str(training_params),
            'Best Other Params': "",
            'Auroc': str(max(combs['auroc'])),
            'Avg_prec': str(max(combs['avg_prec'])),
            'Commentary': commentary,
            'Date': str(datetime.now().strftime("%m/%d/%Y %H:%M:%S")),
            'Results File': "",
            'Experiment Folder Title': exp_results_folder_name,

        }
        write_experiments(exp_results_folder_path.removesuffix(exp_results_folder_name) + "/experiments.csv",
                          row_to_add)

        # Save Best Predictions
        predictions.to_csv(exp_results_folder_path + "/predictions_AUC.csv")

        # Save Model
        model.save_model(exp_results_folder_path + '/model_AUC.txt')

    return combs, best_auc, best_prec


def test_parameter_space_lgb(data_params,
                             test_percentage,
                             model_param_options,
                             patient_col,
                             hour_col,
                             encounter_col,
                             label_col,
                             path_to_results_directory,
                             phenotype,
                             title,
                             description="",
                             commentary=""):
    exp_results_folder_name = str(datetime.now().strftime("%m-%d-%Y %H:%M")) + "-" + title
    exp_results_folder_path = path_to_results_directory + "/" + phenotype + "/" + exp_results_folder_name

    data = data_utils.get_data(**data_params)

    # Split Data
    train_set, test_set = data_utils.make_train_test_splits_by_patients(
        data,
        patient_col=patient_col,
        test_percentage=test_percentage
    )

    # Make dictionaries to store best values/predictions/model
    combs = pd.DataFrame()
    best_auc = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    best_prec = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    # Iterate param space
    baseline_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'verbose': 0,
        'task': 'train',
    }
    args = model_param_options.values()
    keys = list(model_param_options.keys())
    for combination in itertools.product(*args):
        c = {keys[i]: combination[i] for i in range(len(keys))}
        training_params = {**baseline_params, **c}

        train_results, auc_results, prec_results = train(train_set=train_set,
                                                         test_set=test_set,
                                                         label_col=label_col,
                                                         encounter_col=encounter_col,
                                                         patient_col=patient_col,
                                                         hour_col=hour_col,
                                                         model_params=training_params,
                                                         has_weight_col=False,
                                                         path_to_results_directory="",
                                                         data_params_for_logging=data_params,
                                                         title="",
                                                         phenotype=phenotype,
                                                         save=False,
                                                         values_tested="",
                                                         description="",
                                                         commentary="")

        if auc_results['metric'] > best_auc['metric']:
            best_auc = auc_results
        if prec_results['metric'] > best_prec['metric']:
            best_prec = prec_results

        combs = combs.append(train_results, ignore_index=True)

    # Print Results
    print("Max AUROC: " + str(max(combs['auroc'])))
    print("Max Precison: " + str(max(combs['avg_prec'])))

    print("\n# Param values for Max AUROC # \n")

    max_auroc = combs[combs['auroc'] == max(combs['auroc'])]
    pr = max_auroc['Combined Training Params'].iloc[0]
    print(pr)

    print("\n# Param values for Max Average Precision #")
    max_prec = combs[combs['avg_prec'] == max(combs['avg_prec'])]
    pr2 = max_prec['Combined Training Params'].iloc[0]

    # Save Experiment Results
    print("Saving Results to " + exp_results_folder_path)
    os.mkdir(exp_results_folder_path)
    combs.to_csv(exp_results_folder_path + "/experiment_run.csv")
    row_to_add = {
        'Unnamed: 0': '0',
        'Title': title,
        'Description': description,
        'Function Used': 'test_parameter_space_lgb',
        'Values Tested': str(model_param_options),
        'Data Params': str(data_params),
        'Best AUC Params': str(pr),
        'Best Prec Params': str(pr2),
        'Best Other Params': "",
        'Auroc': str(max(combs['auroc'])),
        'Avg_prec': str(max(combs['avg_prec'])),
        'Commentary': commentary,
        'Date': str(datetime.now().strftime("%m/%d/%Y %H:%M:%S")),
        'Experiment Folder Title': exp_results_folder_name,
    }
    write_experiments(exp_results_folder_path.removesuffix(exp_results_folder_name) + "/experiments.csv", row_to_add)

    # Save Best Predictions
    best_auc['predictions'].to_csv(exp_results_folder_path + "/predictions_AUC.csv")
    best_prec['predictions'].to_csv(exp_results_folder_path + "/predictions_PREC.csv")

    # Save Best Model
    best_auc['model'].save_model(exp_results_folder_path + '/model_AUC.txt')
    best_prec['model'].save_model(exp_results_folder_path + '/model_PREC.txt')

    return combs


def gap_experiment_lgb(data_params,
                       gap_hour_options,
                       test_percentage,
                       training_params,
                       patient_col,
                       encounter_col,
                       hour_col,
                       label_col,
                       path_to_results_directory,
                       phenotype,
                       title,
                       description="",
                       commentary=""):
    exp_results_folder_name = str(datetime.now().strftime("%m-%d-%Y %H:%M")) + "-" + title
    exp_results_folder_path = path_to_results_directory + "/" + phenotype + "/" + exp_results_folder_name

    combs = pd.DataFrame()

    best_auc = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    best_prec = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    # Make Dataset
    data = data_utils.get_data(**data_params)
    train_set, test_set = data_utils.make_train_test_splits_by_patients(
        data,
        patient_col=patient_col,
        test_percentage=test_percentage
    )

    for gap_hour in gap_hour_options:

        train_set_with_gap = data_utils.make_gap_before_condition_met(
            df=train_set,
            gap=gap_hour,
            encounter_col_name=encounter_col,
            hour_col_name=hour_col,
            label_col_name=label_col
        )

        train_results, auc_results, prec_results = train(train_set=train_set_with_gap,
                                                         test_set=test_set,
                                                         label_col=label_col,
                                                         encounter_col=encounter_col,
                                                         hour_col=hour_col,
                                                         patient_col=patient_col,
                                                         model_params=training_params,
                                                         has_weight_col=False,
                                                         path_to_results_directory=path_to_results_directory,
                                                         data_params_for_logging=data_params,
                                                         title="",
                                                         phenotype=phenotype,
                                                         save=False,
                                                         values_tested="",
                                                         description="",
                                                         commentary="")
        train_results['gap'] = gap_hour
        if auc_results['metric'] > best_auc['metric']:
            best_auc = auc_results
        if prec_results['metric'] > best_prec['metric']:
            best_prec = prec_results

        combs = combs.append(train_results, ignore_index=True)

    max_auroc = combs[combs['auroc'] == max(combs['auroc'])]
    pr = max_auroc['Combined Training Params'].iloc[0]
    print(pr)

    print("\n# Param values for Max Average Precision #")
    max_prec = combs[combs['avg_prec'] == max(combs['avg_prec'])]
    pr2 = max_prec['Combined Training Params'].iloc[0]
    print(pr2)

    # Save Experiment Results
    os.mkdir(exp_results_folder_path)
    row_to_add = {
        'Unnamed: 0': '0',
        'Title': title,
        'Description': description,
        'Function Used': 'gap_experiment_lgb',
        'Values Tested': 'gap_hours: ' + str(gap_hour_options),
        'Data Params': str(data_params),
        'Best AUC Params': str(pr),
        'Best Prec Params': str(pr2),
        'Best Other Params': "",
        'Auroc': str(max(combs['auroc'])),
        'Avg_prec': str(max(combs['avg_prec'])),
        'Commentary': commentary,
        'Date': str(datetime.now().strftime("%m/%d/%Y %H:%M:%S")),
        'Experiment Folder Title': exp_results_folder_name,
    }
    # Write best outcomes in experiments log
    write_experiments(exp_results_folder_path.removesuffix(exp_results_folder_name) + "/experiments.csv", row_to_add)

    # Save Best Predictions
    best_auc['predictions'].to_csv(exp_results_folder_path + "/predictions_AUC.csv")
    best_prec['predictions'].to_csv(exp_results_folder_path + "/predictions_PREC.csv")

    # Save Best Model
    best_auc['model'].save_model(exp_results_folder_path + '/model_AUC.txt')
    best_prec['model'].save_model(exp_results_folder_path + '/model_PREC.txt')

    # Save Experiment Run
    combs.to_csv(exp_results_folder_path + "/experiment_run.csv")

    return combs


def weighting_boundary_experiment_lgb(
        data_params,
        model_param_options,
        test_percentage,
        patient_col,
        encounter_col,
        label_col,
        hour_col,
        path_to_results_directory,
        phenotype,
        weights,
        decline_ratios,
        title,
        description="",
        commentary=""):
    exp_results_folder_name = str(datetime.now().strftime("%m-%d-%Y %H:%M")) + "-" + title
    exp_results_folder_path = path_to_results_directory + "/" + phenotype + "/" + exp_results_folder_name
    print(weights)
    print(decline_ratios)

    # Make variables to store experiment results
    combs = pd.DataFrame()
    best_auc = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    best_prec = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    # Make Dataset
    data = data_utils.get_data(**data_params)
    train_set, test_set = data_utils.make_train_test_splits_by_patients(
        data,
        patient_col=patient_col,
        test_percentage=test_percentage
    )

    for i in range(len(weights)):
        for j in range(len(decline_ratios)):
            print("Weight = " + str(weights[i]))
            print("Decline = " + str(decline_ratios[j]))

            train_set_with_weight = data_utils.make_weight_column_distribution(
                train_set,
                max_weight=weights[i],
                increment=weights[i] * decline_ratios[j],
                label_col=label_col,
                encounter_col=encounter_col,
                hour_col=hour_col
            )
            print("Train Set Average Weight = " + str(train_set_with_weight['weight'].mean()))

            args = model_param_options.values()
            keys = list(model_param_options.keys())

            for combination in itertools.product(*args):

                model_params = {keys[i]: combination[i] for i in range(len(keys))}
                train_results, auc_results, prec_results = train(train_set=train_set_with_weight,
                                                                 test_set=test_set,
                                                                 label_col=label_col,
                                                                 encounter_col=encounter_col,
                                                                 hour_col=hour_col,
                                                                 patient_col=patient_col,
                                                                 model_params=model_params,
                                                                 has_weight_col=True,
                                                                 path_to_results_directory=path_to_results_directory,
                                                                 data_params_for_logging=data_params,
                                                                 title="",
                                                                 phenotype=phenotype,
                                                                 save=False,
                                                                 values_tested="",
                                                                 description="",
                                                                 commentary="")

                train_results['weight'] = i
                train_results['decline'] = j
                combs = combs.append(train_results, ignore_index=True)
                if auc_results['metric'] > best_auc['metric']:
                    best_auc = auc_results
                if prec_results['metric'] > best_prec['metric']:
                    best_prec = prec_results

    max_auroc = combs[combs['auroc'] == max(combs['auroc'])]
    pr = max_auroc['Combined Training Params'].iloc[0]
    print(pr)

    print("\n# Param values for Max Average Precision #")
    max_prec = combs[combs['avg_prec'] == max(combs['avg_prec'])]
    pr2 = max_prec['Combined Training Params'].iloc[0]
    print(pr2)

    best_weight = max_auroc['weight'].iloc[0]
    best_decline = max_auroc['decline'].iloc[0]

    # Save Experiment Results
    os.mkdir(exp_results_folder_path)
    combs.to_csv(exp_results_folder_path + "/experiment_run.csv")

    row_to_add = {
        'Unnamed: 0': '0',
        'Title': title,
        'Description': description,
        'Function Used': 'weighting_experiment_lgb',
        'Values Tested': {
            'weights': str(weights),
            'declines': str(decline_ratios)},
        'Data Params': str(data_params),
        'Best AUC Params': str(pr),
        'Best Prec Params': str(pr2),
        'Best Other Params': str({
            'weight': best_weight, 'decline': best_decline,
        }),
        'Auroc': str(max(combs['auroc'])),
        'Avg_prec': str(max(combs['avg_prec'])),
        'Commentary': commentary,
        'Date': str(datetime.now().strftime("%m/%d/%Y %H:%M:%S")),
        'Experiment Folder Title': exp_results_folder_name,
    }
    # Write best outcomes in experiments log
    write_experiments(exp_results_folder_path.removesuffix(exp_results_folder_name) + "/experiments.csv", row_to_add)

    # Save Best Predictions
    best_auc['predictions'].to_csv(exp_results_folder_path + "/predictions_AUC.csv")
    best_prec['predictions'].to_csv(exp_results_folder_path + "/predictions_PREC.csv")

    # Save Best Model
    best_auc['model'].save_model(exp_results_folder_path + '/model_AUC.txt')
    best_prec['model'].save_model(exp_results_folder_path + '/model_PREC.txt')
    return combs


def weighting_experiment_lgb(
        data_params,
        model_param_options,
        test_percentage,
        patient_col,
        encounter_col,
        label_col,
        hour_col,
        path_to_results_directory,
        phenotype,
        weights,
        title,
        description="",
        commentary=""):
    exp_results_folder_name = str(datetime.now().strftime("%m-%d-%Y %H:%M")) + "-" + title
    exp_results_folder_path = path_to_results_directory + "/" + phenotype + "/" + exp_results_folder_name
    print(weights)

    # Make variables to store experiment results
    combs = pd.DataFrame()
    best_auc = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    best_prec = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    # Make Dataset
    data = data_utils.get_data(**data_params)
    train_set, test_set = data_utils.make_train_test_splits_by_patients(
        data,
        patient_col=patient_col,
        test_percentage=test_percentage
    )

    for i in range(len(weights)):
        print("Weight = " + str(weights[i]))

        train_set_with_weight = data_utils.make_weight_column(
            train_set,
            pos_weight=weights[i],
            label_col=label_col,
            encounter_col=encounter_col,
            hour_col=hour_col
        )
        print("Train Set Average Weight = " + str(train_set_with_weight['weight'].mean()))

        args = model_param_options.values()
        keys = list(model_param_options.keys())

        for combination in itertools.product(*args):

            model_params = {keys[i]: combination[i] for i in range(len(keys))}
            train_results, auc_results, prec_results = train(train_set=train_set_with_weight,
                                                             test_set=test_set,
                                                             label_col=label_col,
                                                             encounter_col=encounter_col,
                                                             hour_col=hour_col,
                                                             patient_col=patient_col,
                                                             model_params=model_params,
                                                             has_weight_col=True,
                                                             path_to_results_directory=path_to_results_directory,
                                                             data_params_for_logging=data_params,
                                                             title="",
                                                             phenotype=phenotype,
                                                             save=False,
                                                             values_tested="",
                                                             description="",
                                                             commentary="")

            train_results['weight'] = i
            combs = combs.append(train_results, ignore_index=True)
            if auc_results['metric'] > best_auc['metric']:
                best_auc = auc_results
            if prec_results['metric'] > best_prec['metric']:
                best_prec = prec_results

    max_auroc = combs[combs['auroc'] == max(combs['auroc'])]
    pr = max_auroc['Combined Training Params'].iloc[0]
    print(pr)

    print("\n# Param values for Max Average Precision #")
    max_prec = combs[combs['avg_prec'] == max(combs['avg_prec'])]
    pr2 = max_prec['Combined Training Params'].iloc[0]
    print(pr2)

    best_weight = max_auroc['weight'].iloc[0]

    # Save Experiment Results
    os.mkdir(exp_results_folder_path)
    combs.to_csv(exp_results_folder_path + "/experiment_run.csv")

    row_to_add = {
        'Unnamed: 0': '0',
        'Title': title,
        'Description': description,
        'Function Used': 'class_weighting_experiment_lgb',
        'Values Tested': {
            'weights': str(weights),
        },
        'Data Params': str(data_params),
        'Best AUC Params': str(pr),
        'Best Prec Params': str(pr2),
        'Best Other Params': str({
            'weight': best_weight,
        }),
        'Auroc': str(max(combs['auroc'])),
        'Avg_prec': str(max(combs['avg_prec'])),
        'Commentary': commentary,
        'Date': str(datetime.now().strftime("%m/%d/%Y %H:%M:%S")),
        'Experiment Folder Title': exp_results_folder_name,
    }
    # Write best outcomes in experiments log
    write_experiments(exp_results_folder_path.removesuffix(exp_results_folder_name) + "/experiments.csv", row_to_add)

    # Save Best Predictions
    best_auc['predictions'].to_csv(exp_results_folder_path + "/predictions_AUC.csv")
    best_prec['predictions'].to_csv(exp_results_folder_path + "/predictions_PREC.csv")

    # Save Best Model
    best_auc['model'].save_model(exp_results_folder_path + '/model_AUC.txt')
    best_prec['model'].save_model(exp_results_folder_path + '/model_PREC.txt')
    return combs


def backfill_experiment_lgb(data_params,
                            test_percentage,
                            backfill_options,
                            model_params,
                            patient_col,
                            encounter_col,
                            hour_col,
                            label_col,
                            path_to_results_directory,
                            phenotype,
                            title,
                            description="",
                            commentary=""):
    exp_results_folder_name = str(datetime.now().strftime("%m-%d-%Y %H:%M")) + "-" + title
    exp_results_folder_path = path_to_results_directory + "/" + phenotype + "/" + exp_results_folder_name

    # Get Base Dataset

    data = data_utils.read_in_chunks(data_params['data_path'])
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # Make Storage Variables
    combs = pd.DataFrame()
    best_auc = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    best_prec = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    for backfill in backfill_options:
        # apply backfill
        data_with_backfill = data_utils.apply_backfill(
            df=data,
            hours=backfill,
            encounter_col_name=encounter_col,
            hour_col_name=hour_col,
            label_col_name=label_col)

        # Split
        train_set, test_set = data_utils.make_train_test_splits_by_patients(
            data_with_backfill,
            patient_col=patient_col,
            test_percentage=test_percentage
        )

        # Train
        train_results, auc_results, prec_results = train(train_set=train_set,
                                                         test_set=test_set,
                                                         label_col=label_col,
                                                         encounter_col=encounter_col,
                                                         hour_col=hour_col,
                                                         patient_col=patient_col,
                                                         model_params=model_params,
                                                         has_weight_col=False,
                                                         path_to_results_directory=path_to_results_directory,
                                                         data_params_for_logging=data_params,
                                                         title="",
                                                         phenotype=phenotype,
                                                         save=False,
                                                         values_tested="",
                                                         description="",
                                                         commentary="")

        train_results['backfill'] = backfill
        combs = combs.append(train_results, ignore_index=True)
        if auc_results['metric'] > best_auc['metric']:
            best_auc = auc_results
        if prec_results['metric'] > best_prec['metric']:
            best_auc = prec_results

    os.mkdir(exp_results_folder_path)
    combs.to_csv(exp_results_folder_path + "/experiment_run.csv")

    print("\n# Param values for Max AUROC #")
    max_auroc = combs[combs['auroc'] == max(combs['auroc'])]
    pr = max_auroc['Combined Training Params'].iloc[0]
    print(pr)

    print("\n# Param values for Max Average Precision #")
    max_prec = combs[combs['avg_prec'] == max(combs['avg_prec'])]
    pr2 = max_prec['Combined Training Params'].iloc[0]
    print(pr2)

    best_backfill = max_auroc['backfill'].iloc[0]

    # Save Experiment Results
    row_to_add = {
        'Unnamed: 0': '0',
        'Title': title,
        'Description': description,
        'Function Used': 'backfill_experiment_lgb',
        'Values Tested': 'backfill_options: ' + str(backfill_options),
        'Data Params': str(data_params),
        'Best AUC Params': str(pr),
        'Best Prec Params': str(pr2),
        'Best Other Params': best_backfill,
        'Auroc': str(max(combs['auroc'])),
        'Avg_prec': str(max(combs['avg_prec'])),
        'Commentary': commentary,
        'Date': str(datetime.now().strftime("%m/%d/%Y %H:%M:%S")),
        'Experiment Folder Title': exp_results_folder_name,
    }
    # Write best outcomes in experiments log
    write_experiments(exp_results_folder_path.removesuffix(exp_results_folder_name) + "/experiments.csv", row_to_add)

    # Save Best Predictions
    best_auc['predictions'].to_csv(exp_results_folder_path + "/predictions_AUC.csv")
    best_prec['predictions'].to_csv(exp_results_folder_path + "/predictions_PREC.csv")

    # Save Best Model
    best_auc['model'].save_model(exp_results_folder_path + '/model_AUC.txt')
    best_prec['model'].save_model(exp_results_folder_path + '/model_PREC.txt')

    return combs


def prevalence_experiment_lgb(
        data_params,
        test_percentage,
        prevalence_options,
        model_params,
        patient_col,
        hour_col,
        encounter_col,
        label_col,
        path_to_results_directory,
        phenotype,
        title,
        description="",
        commentary=""):
    # Make Paths
    exp_results_folder_name = str(datetime.now().strftime("%m-%d-%Y %H:%M")) + "-" + title
    exp_results_folder_path = path_to_results_directory + "/" + phenotype + "/" + exp_results_folder_name

    # Make variables to store results
    combs = pd.DataFrame()
    best_auc = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }
    best_prec = {
        'metric': 0,
        'predictions': pd.DataFrame(),
        'model': lgb.Booster(train_set=lgb.Dataset(pd.Series([0.0, 0.0])))
    }

    # Get Base Dataset
    data = data_utils.get_data(**data_params)

    # Split
    train_set, test_set = data_utils.make_train_test_splits_by_patients(
        data,
        patient_col=patient_col,
        test_percentage=test_percentage
    )

    # Iterate options
    for prev in prevalence_options:
        # Apply prevalence to test set
        train_set_with_prevalence = data_utils.sample_for_class_balance(
            data=train_set,
            label_column=label_col,
            prevalence=prev,
            encounter_col=encounter_col)

        # Train
        train_results, auc_results, prec_results = train(train_set=train_set_with_prevalence,
                                                         test_set=test_set,
                                                         label_col=label_col,
                                                         encounter_col=encounter_col,
                                                         hour_col=hour_col,
                                                         patient_col=patient_col,
                                                         model_params=model_params,
                                                         has_weight_col=False,
                                                         path_to_results_directory=path_to_results_directory,
                                                         data_params_for_logging=data_params,
                                                         title="",
                                                         phenotype=phenotype,
                                                         save=False,
                                                         values_tested="",
                                                         description="",
                                                         commentary="")

        train_results['prevalence'] = prev
        combs = combs.append(train_results, ignore_index=True)

        if auc_results['metric'] > best_auc['metric']:
            best_auc = auc_results
            best_prev = prev
        if prec_results['metric'] > best_prec['metric']:
            best_auc = prec_results
    # Save
    os.mkdir(exp_results_folder_path)
    combs.to_csv(exp_results_folder_path + "/experiment_run.csv")

    max_auroc = combs[combs['auroc'] == max(combs['auroc'])]
    pr = max_auroc['Combined Training Params'].iloc[0]
    print(pr)

    print("\n# Param values for Max Average Precision #")
    max_prec = combs[combs['avg_prec'] == max(combs['avg_prec'])]
    pr2 = max_prec['Combined Training Params'].iloc[0]
    print(pr2)

    print("\n# Best prevalence for AUROC #")
    best_auc_prev = combs[combs['auroc'] == max(combs['auroc'])]['prevalence'].iloc[0]
    print(best_auc_prev)
    print("\n# Best prevalence for PREC #")
    best_prec_prev = combs[combs['avg_prec'] == max(combs['avg_prec'])]['prevalence'].iloc[0]
    print(best_prec_prev)

    # Save Experiment Results
    row_to_add = {
        'Unnamed: 0': '0',
        'Title': title,
        'Description': description,
        'Function Used': 'prevalence_experiment_lgb',
        'Values Tested': 'prevalence options: ' + str(prevalence_options),
        'Data Params': str(data_params),
        'Best AUC Params': str(pr),
        'Best Prec Params': str(pr2),
        'Best Other Params': {
            "Best AUC Prev:": best_auc_prev,
            "Best PREC Prev:": best_prec_prev,
        },
        'Auroc': str(max(combs['auroc'])),
        'Avg_prec': str(max(combs['avg_prec'])),
        'Commentary': commentary,
        'Date': str(datetime.now().strftime("%m/%d/%Y %H:%M:%S")),
        'Experiment Folder Title': exp_results_folder_name,
    }
    # Write best outcomes in experiments log
    write_experiments(exp_results_folder_path.removesuffix(exp_results_folder_name) + "/experiments.csv", row_to_add)

    # Save Best Predictions
    best_auc['predictions'].to_csv(exp_results_folder_path + "/predictions_AUC.csv")
    best_prec['predictions'].to_csv(exp_results_folder_path + "/predictions_PREC.csv")

    # Save Best Model
    best_auc['model'].save_model(exp_results_folder_path + '/model_AUC.txt')
    best_prec['model'].save_model(exp_results_folder_path + '/model_PREC.txt')

    return combs
