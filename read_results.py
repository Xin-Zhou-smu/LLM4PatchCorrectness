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
from collections import Counter

paths = ['./checked_data_cross_tool_enhanced_results_starcoder_7b/bigcode/starcoderbase-7b/direct/']
projects = ['ACS', 'Arja', 'AVATAR', 'CapGen',  'Cardumen', 'DynaMoth', 'FixMiner', 'GenProg', 'Jaid', 'jGenProg', 'jKali', 'jMutRepair', 'Kali', 'kPAR', 'Nopol', 'RSRepair', 'SequenceR', 'SimFix', 'SketchFix', 'SOFix', 'TBar']


num_smaples_projects = []
all_metrics1 = {}
template = 0
topk = 10
simthreshold=0.9

all_llm_preds, all_llm_labels = [],[]

for project in projects:

        ex_data = pd.read_csv('./data_checked/patch_cor/' + project + '_test_v1.csv', header=None)
        num_smaples_projects.append(len(list(ex_data.iloc[:,0])))

        predictions_from_all_models = []
        prediction_scores_from_all_models = []

        missing_file = False
        path_idx = 0

        for path in paths:
            for option in ["bug-trace-testcase-coverage-similar"]:

                if 'codegen2' in path or 'bloom-1b7' in path:
                    maxlength = 2000
                elif 'starcoder' in path or 'codellama' in path:
                    maxlength = 4000
                elif 'codeparrot' in path:
                    maxlength = 1000
                file_ = path + 'patch_'+project+'/test-template='+str(template)+'-topk_example='+str(topk)+'-sim_threshold='+str(simthreshold)+'-max_length='+str(maxlength)+'-'+option+'.pkl'

                if not os.path.exists(file_):
                    print(file_)
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


                    raw_data = pd.read_csv('./data_checked/patch_cor/' + project + '_test_v1.csv', header=None)

                    labels = list(raw_data.iloc[:,0])



                    real_pred = pred

                    predictions_from_all_models.append( (real_pred, labels) )
                    prediction_scores_from_all_models.append(  (pred_score, labels)  )

                    from collections import Counter

                    # print(Counter(labels))
                    # labels = [0 if int(l) == 1 else 1 for l in labels]  ##为了使得correct是1类，所以需要转换一下labels
                    # print(Counter(labels), len(labels))
                    # real_pred = [0 if int(l) == 1 else 1 for l in real_pred]
                    # raw_pred = [1-p for p in raw_pred]


                    try:
                        # auc_score = round(roc_auc_score(y_true=labels, y_score=real_pred), 3)
                        auc_score = round(roc_auc_score(y_true=labels, y_score=raw_pred), 3)
                    except:
                        auc_score = 'null'

                    print(project, Counter(labels))

                    accuracy_ = round(accuracy_score(y_true=labels, y_pred=real_pred), 3)
                    prec = round(precision_score(y_true=labels, y_pred=real_pred), 3)
                    recal = round(recall_score(y_true=labels, y_pred=real_pred), 3)
                    f1 = round(f1_score(y_true=labels, y_pred=real_pred), 3)
                    ma_f1 = round(f1_score(y_true=labels, y_pred=real_pred, average='macro'), 3)
                    coef = round(matthews_corrcoef(labels, real_pred), 3)
                    print("{}, {}, ONE MODEL ACC:{}  F1-Score:{}  M-F1:{}  Prec:{} Recall:{}  MCC: {} AUC:{}" .format(path.split('/')[-3:-2][0], project, accuracy_, f1, ma_f1, prec, recal, coef, auc_score))
                    print()










