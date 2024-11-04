import sklearn.metrics as skmetr


def _convert_probas_to_binary(probas):
    return 1.0*(probas[:,1] > 0.5)

def accuracy_score(probas, ytrue):
    return skmetr.accuracy_score(ytrue, _convert_probas_to_binary(probas))

def precision_score(probas, ytrue):
    return skmetr.precision_score(ytrue, _convert_probas_to_binary(probas))

def recall_score(probas, ytrue):
    return skmetr.recall_score(ytrue, _convert_probas_to_binary(probas))

def f1_score(probas, ytrue):
    return skmetr.f1_score(ytrue, _convert_probas_to_binary(probas))

def roc_auc_score(probas, ytrue):
    return skmetr.roc_auc_score(ytrue, probas[:,1])


def compute_metrics_list(metric_list, probas, ytrue):
    res_dict = {}
    for m in metric_list:
        metric_func = globals()[m]
        res_dict[m] = metric_func(probas, ytrue)
    return res_dict


