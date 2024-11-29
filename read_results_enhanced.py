
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, \
    roc_auc_score, average_precision_score, precision_recall_curve, matthews_corrcoef
import pandas as pd
import pickle
import numpy as np
import math, os
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, \
    roc_auc_score, average_precision_score, precision_recall_curve, matthews_corrcoef
import pandas as pd
import pickle
import numpy as np
import math
from sklearn.metrics import confusion_matrix


paths = ['results/bigcode/starcoderbase-7b/direct/']

projects = ['ACS', 'Arja', 'AVATAR', 'CapGen',  'Cardumen', 'DynaMoth', 'FixMiner', 'GenProg', 'HDRepair', 'Jaid', 'jGenProg', 'jKali', 'jMutRepair', 'Kali', 'kPAR', 'Nopol', 'RSRepair', 'SequenceR', 'SimFix', 'SketchFix', 'SOFix', 'TBar']


num_smaples_projects = []
all_metrics1 = {}
template = 0
topk = 10
simthreshold=0.9

for project in projects:

        ex_data = pd.read_csv('data_checked/patch_cor/' + project + '_test_v1.csv', header=None)
        num_smaples_projects.append(len(list(ex_data.iloc[:,0])))

        predictions_from_all_models = []
        prediction_scores_from_all_models = []

        missing_file = False
        path_idx = 0

        for path in paths:
            for option in ["bug-trace-testcase-coverage-similar"]:
                if 'starcoder' in path:
                    maxlength = 4000
                file_ = path + 'patch_'+project+'/test-template='+str(template)+'-topk_example='+str(topk)+'-sim_threshold='+str(simthreshold)+'-max_length='+str(maxlength)+'-'+option+'.pkl'

                if not os.path.exists(file_):
                    missing_file = True
                else:
                    f = open(file_, 'rb')
                    data = pickle.load(f)

                    list1 = data[0]
                    list2 = data[1]

                    pred, raw_pred = [],[]
                    pred_score = []

                    for i in range(len(list1)):
                        if list1[i] >= list2[i]:
                            pred.append(1)
                        else:
                            pred.append(0)

                        def softmax(x):
                            """Compute the softmax of vector x."""
                            exp_x = np.exp(x)
                            softmax_x = exp_x / np.sum(exp_x)
                            return list(softmax_x)
                        raw_pred.append(softmax([list1[i], list2[i]])[0])

                        pred_score.append( (list1[i], list2[i]) )


                    raw_data = pd.read_csv('data_checked/patch_cor/' + project + '_test_v1.csv', header=None)

                    labels = list(raw_data.iloc[:,0])
                    real_pred = pred

                    predictions_from_all_models.append( (real_pred, labels) )
                    prediction_scores_from_all_models.append(  (pred_score, labels)  )

                    from collections import Counter

                    try:
                        auc_score = round(roc_auc_score(y_true=labels, y_score=raw_pred), 3)
                    except:
                        auc_score = 'null'


                    accuracy_ = round(accuracy_score(y_true=labels, y_pred=real_pred), 3)
                    prec = round(precision_score(y_true=labels, y_pred=real_pred), 3)
                    recal = round(recall_score(y_true=labels, y_pred=real_pred), 3)
                    f1 = round(f1_score(y_true=labels, y_pred=real_pred), 3)
                    ma_f1 = round(f1_score(y_true=labels, y_pred=real_pred, average='macro'), 3)
                    coef = round(matthews_corrcoef(labels, real_pred), 3)
                    tn, fp, fn, tp = confusion_matrix(labels, real_pred).ravel()
                    
                    if (fp+tn) == 0:
                        fpr = 0
                    else:
                        fpr = fp / (fp + tn)

                    if (fn + tp) == 0:
                        fnr = 0
                    else:
                        fnr = fn / (fn + tp)
                    
                    print("{}, {}, ONE MODEL ACC:{}  F1-Score:{}  M-F1:{}  Prec:{} Recall:{}  FPR: {} FNR:{} AUC:{}".format(path.split('/')[-3:-2][0], project, accuracy_, f1, ma_f1, prec, recal, fpr, fnr, auc_score))
                    
                path_idx = option
                if path_idx not in all_metrics1:
                    all_metrics1[path_idx] = []
                    all_metrics1[path_idx].append((accuracy_, f1, ma_f1, prec, recal, fpr, fnr, auc_score))
                else:
                    all_metrics1[path_idx].append((accuracy_, f1, ma_f1, prec, recal, fpr, fnr, auc_score))
                
 


def report_average(list_l, weights=None):
    metrics_1, metrics_2,metrics_3,metrics_4,metrics_5,metrics_6,metrics_7,metrics_8 =[],[],[],[],[],[],[],[]
    print(len(list_l))
    l_id = 0
    for l in list_l:
        metric_1, metric_2, metric_3, metric_4, metric_5, metric_6, metric_7, metric_8 = l
        if weights is None:
            metrics_1.append(metric_1)
            metrics_2.append(metric_2)
            metrics_3.append(metric_3)
            metrics_4.append(metric_4)
            metrics_5.append(metric_5)
            metrics_6.append(metric_6)
            metrics_7.append(metric_7)
            if metric_8 != 'null':
                metrics_8.append(metric_8)
        else:
            print('weighted avg.')
            metrics_1.append(metric_1*num_smaples_projects[l_id])
            metrics_2.append(metric_2*num_smaples_projects[l_id])
            metrics_3.append(metric_3*num_smaples_projects[l_id])
            metrics_4.append(metric_4*num_smaples_projects[l_id])
            metrics_5.append(metric_5*num_smaples_projects[l_id])
            metrics_6.append(metric_6*num_smaples_projects[l_id])
            metrics_7.append(metric_7 * num_smaples_projects[l_id])
            if metric_8 != 'null':
                metrics_8.append(metric_8 * num_smaples_projects[l_id])
        l_id +=1
    if weights is None:
        return round(sum(metrics_1)/len(metrics_1), 3), round(sum(metrics_2)/len(metrics_2), 3), round(sum(metrics_3)/len(metrics_3), 3), \
                round(sum(metrics_4) / len(metrics_4), 3), round(sum(metrics_5)/len(metrics_5), 3), round(sum(metrics_6)/len(metrics_6), 3), round(sum(metrics_7)/len(metrics_7), 3), round(sum(metrics_8)/len(metrics_8), 3)
    else:
        return round(sum(metrics_1) / sum(num_smaples_projects), 3), round(sum(metrics_2) / sum(num_smaples_projects), 3), round(sum(metrics_3) / sum(num_smaples_projects), 3), \
            round(sum(metrics_4) / sum(num_smaples_projects), 3), round(sum(metrics_5) / sum(num_smaples_projects), 3), round(sum(metrics_6) / sum(num_smaples_projects), 3), round(sum(metrics_7) / sum(num_smaples_projects), 3),  round(sum(metrics_8) / (sum(num_smaples_projects)-9), 3)

print(path)
print('\naverage on projects')
for key in all_metrics1:
    res = report_average(all_metrics1[key])
    string_lists = ['ACC:', 'F1-Score:', 'M-F1:', 'Prec:', 'Recall:', 'FPR:', 'FNR', 'AUC']
    report_string = ''
    for ijij in range(len(res)):
        report_string += string_lists[ijij] + " " + str(res[ijij]) + " "
    print(report_string)




