## Train models and optimize hyperparameters.
## We use Optuna to train the models.
## Usage: train_models.py <hyperparamter file> <Model class> <--retrain>
import optuna
import os, sys
import json
from optuna.trial import TrialState
from datasets import DatasetDict, Dataset
import torch
import numpy as np
import argparse
import itertools

import code.classification_models as cls_model
from code.dataset import load_gtc, load_ptsd, load_counsel, load_incels
import code.compute_metrics as metrics

## GLOBAL CONFIGURATION
USE_MODEL = ""
hyperparameters = ""
genocide_ds = load_gtc("Dataset_GTC-V2.csv")
OPTIM_METRIC = "accuracy_score" # Metric that is optimized.
ALL_METRICS = ["accuracy_score", "f1_score", "precision_score", "recall_score", "roc_auc_score"]
LOG_PATH = "logs_{}"
LOG = "hyperparams{}.json" # File name pattern for hyperparameter search log files
LOG_MODEL = "models/{}.pt" # Log hyperparamters
PREDS_MODEL = "models/{}-preds.pt"  # Log prediction template
USE_DS = None
USE_DEVICE = None

def arg_parse():
    # add arguments 
    ## There are two modes for this script: 1st hyperparameter optimization mode (retrain == False). 2nd Train models with fixed hyperparameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparameter_file', type=str, help='hyperparameter file to use')
    parser.add_argument('classification_model', type=str, help='which model to use, e.g. BERTmodel, OpenAImodel, (class name of src.classification_models)')
    parser.add_argument('dataset', type=str, help='dataset to use, e.g., GTC, PTSD')
    parser.add_argument('--device', type=str, help='which device to run on', default='cuda')
    parser.add_argument('--retrain', type=bool, help='if passed, only compute the metrics and train retrain_iters model runs, no hyperparameter opt', default=False)
    parser.add_argument('--retrain_iters', type=int, help='number of models to train in retrain mode.', default=5)
    args = parser.parse_args()
    return args

def log_metrics(init_params, train_params, metrics):
    os.makedirs(LOG_DIR, exist_ok = True)
    if os.path.exists(LOG_DIR + "/" + LOG.format(USE_MODEL)):
        res_list = json.load(open(LOG_DIR + "/" + LOG.format(USE_MODEL)))
    else:
        res_list = []
    res_list.append([dict(**init_params, **train_params), metrics])
    json.dump(res_list, open(LOG_DIR + "/" + LOG.format(USE_MODEL), "w"))

def savemodel(myobj, params, preds=None, suffix=""):
    os.makedirs(LOG_DIR + "/models", exist_ok = True)
    mystr = ""
    for k, v in params.items():
        if k == "device":
            continue
        if mystr != "":
            mystr += "_"
        mystr +=str(k) + "-" + str(v)
    myobj.save(LOG_DIR + "/" + LOG_MODEL.format(USE_MODEL + "_" + mystr + suffix))
    if preds is not None:
        torch.save(preds, LOG_DIR + "/" + PREDS_MODEL.format(USE_MODEL + "_" + mystr + suffix))

def run_model(trial):
    my_params = hyperparameters[USE_MODEL]
    ## Get values
    init_dict = {}
    train_dict = {}
    for param in my_params:
        val = None
        if "options" in param:
            val = trial.suggest_categorical(param["name"], list([tuple(p) if isinstance(p, list) else p for p in param["options"]]))
        if "logrange" in param:
            val = trial.suggest_float(param["name"], param["logrange"][0], param["logrange"][1], log=True)
        if "floatrange" in param:
            val = trial.suggest_float(param["name"], param["logrange"][0], param["logrange"][1])
        if "intrange" in param:
            val = trial.suggest_int(param["name"], param["intrange"][0], param["intrange"][1])
        if param["at_init"]:
            init_dict[param["name"]] = val
        else:
            train_dict[param["name"]] = val

    init_dict["device"] = USE_DEVICE ## Pass device at init
    print(init_dict, train_dict)
    myobjcls = getattr(cls_model, USE_MODEL)
    myobj = myobjcls(**init_dict)
    myobj.train(USE_DS, **train_dict)
    ## Test phase
    probas = myobj.predict_proba(USE_DS["test"]["text"])
    all_metrics = metrics.compute_metrics_list(ALL_METRICS, probas, USE_DS["test"]["label"])
    log_metrics(init_dict, train_dict, all_metrics)
    savemodel(myobj, dict(**init_dict, **train_dict), probas if USE_MODEL=="OpenAImodel" else None)
    return all_metrics[OPTIM_METRIC]

def train_model(params_use, run_id, use_ds, device, dataset_name):
    """ Load hyperparameter configuration and train a model. """
    torch.manual_seed(49*run_id)
    np.random.seed(49*run_id)
    my_params = hyperparameters[USE_MODEL]
    ## Get values
    init_dict = {}
    train_dict = {}
    for param in my_params:
        if param["at_init"]:
            p = params_use[param["name"]]
            init_dict[param["name"]] = tuple(p) if isinstance(p, list) else p
        else:
            p = params_use[param["name"]]
            train_dict[param["name"]] = tuple(p) if isinstance(p, list) else p
    init_dict["device"] = device ## Pass device at init (do not used loaded one)
    print(init_dict, train_dict)
    myobjcls = getattr(cls_model, USE_MODEL)
    myobj = myobjcls(**init_dict)
    myobj.train(use_ds, **train_dict)
    ## Test phase
    probas = myobj.predict_proba(use_ds["test"]["text"])
    all_metrics = metrics.compute_metrics_list(ALL_METRICS, probas, use_ds["test"]["label"])
    savemodel(myobj, dict(**init_dict, **train_dict), probas, suffix= f"_{dataset_name}_r{run_id}")
    return all_metrics


def get_dataset(dataset_name: str, seed=None):
    if dataset_name == "GTC":
        return load_gtc("Dataset_GTC-V2.csv", seed = seed if seed else 1) ## Default seed is 1 here
    elif dataset_name == "GTC-1000":
        return load_gtc("Dataset_GTC-V2.csv", seed = seed if seed is not None else 0, sn_train=1000) 
    elif dataset_name == "PTSD":
        return load_ptsd("final_df_ptsd_1200_training.csv", seed = seed if seed is not None else 0) ## Default seed 0 here.
    elif dataset_name == "Counseling":
        return load_counsel("final_df_counseling_1200_training.csv", seed = seed if seed is not None else 0) ## Default seed 0 here.
    elif dataset_name == "Incels":
        return load_incels("final_df_incels_300_test.csv", seed = seed if seed is not None else 0)
    elif dataset_name == "All":
        gtc = load_gtc("Dataset_GTC-V2.csv", seed = seed if seed else 1) ## Default seed is 1 here
        ptsd = load_ptsd("final_df_ptsd_1200_training.csv", seed = seed if seed is not None else 0)
        counsel = load_counsel("final_df_counseling_1200_training.csv", seed = seed if seed is not None else 0)
        dslist = [gtc, ptsd, counsel]
        xdict = {}
        ydict = {}
        for mode in ["train", "test"]:
            xdict[mode] = list(itertools.chain.from_iterable([ds[mode]["text"] for ds in dslist]))
            ydict[mode] = list(itertools.chain.from_iterable([ds[mode]["label"] for ds in dslist]))
        all_ds = Dataset.from_dict({"text": xdict["train"], "label":  torch.tensor(ydict["train"], dtype=torch.long)})
        all_ds_test = Dataset.from_dict({"text": xdict["test"], "label":  torch.tensor(ydict["test"], dtype=torch.long)})
        return DatasetDict({"train": all_ds, "test": all_ds_test})
    else:
        raise ValueError(f"Unknown dataset {dataset_name}.")


def get_best_params(model_cls: str, dataset_name: str):
    res = json.load(open(f"{LOG_DIR}/best_params_{model_cls}.json"))
    if model_cls == "OpenAImodel":
        res["target_model"] = "gpt-3.5-turbo" ## Hack to not use GPT-4 repetetively atm
        print("Changing target model to gpt-3.5-turbo.")
    return res

if __name__ == "__main__":
    args = arg_parse()
    if len(sys.argv) < 3:
        print("Pass at least two arguments.")
        exit(0)

    hyperparameters = json.load(open(args.hyperparameter_file))
    USE_MODEL = args.classification_model
    USE_DEVICE = args.device
    do_retrain = args.retrain
    LOG_DIR = LOG_PATH.format(args.dataset)
    print(LOG_DIR)
    if do_retrain: ## Retrain best models for error bars (random splits of data)
        params_use = get_best_params(USE_MODEL, args.dataset)
        metricslist = []
        for run in range(args.retrain_iters):
            use_ds = get_dataset(args.dataset, seed=run*49)
            metricslist.append(train_model(params_use, run, use_ds, USE_DEVICE, args.dataset))
        json.dump(metricslist, open(f"{LOG_DIR}/rerun_metrics_{USE_MODEL}.json", "w"))
    else: ## Perform hyperparameter tuning on common train/test split
        USE_DS = get_dataset(args.dataset)
        if USE_MODEL == "OpenAImodel": ## Do a systematic study of hyperparamteres
            search_space = {"target_model": hyperparameters[USE_MODEL][0]["options"]}
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))
        else: ## Do a randomized study
            study = optuna.create_study(direction="maximize")
        study.optimize(run_model, n_trials=50, timeout=None)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        json.dump(trial.params, open(f"{LOG_DIR}/best_params_{USE_MODEL}.json", "w"))
