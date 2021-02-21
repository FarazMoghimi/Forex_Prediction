from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
import codes.prediction.io as io
import numpy as np


def add_column_score(true_data, pred_data, column_name):
    return {
        column_name+'_accuracy': accuracy_score(true_data, pred_data),
        column_name+'_precision': precision_score(true_data, pred_data),
        column_name+'_recall': recall_score(true_data, pred_data),
        column_name+'_f1': f1_score(true_data, pred_data),
        column_name + '_confusion': confusion_matrix(true_data, pred_data)
    }


def get_all_scores_name():
    column_names = io.get_score_columns()
    scores_name = ['all_accuracy', 'all_precision', 'all_recall', 'all_f1']
    for column_name in column_names:
        scores_name.extend([
            column_name + '_accuracy',
            column_name + '_precision',
            column_name + '_recall',
            column_name + '_f1',
            column_name + '_confusion'
        ])
    return scores_name


def get_all_scores(y_data, y_pred, pd_x_data):
    column_names = io.get_score_columns()
    all_scores = add_column_score(y_data, y_pred, 'all')
    for column_name in column_names:
        pred_data = np.array(pd_x_data[column_name].values)
        pred_data = pred_data.astype('int')
        score = add_column_score(y_data, pred_data, column_name)
        all_scores = {**all_scores, **score}
    return all_scores
