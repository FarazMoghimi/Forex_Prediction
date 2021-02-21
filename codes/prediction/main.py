import codes.prediction.io as io
import codes.prediction.svm_predictor as svm_predictor
import numpy as np
import pandas as pd
import os

svm = svm_predictor.SVM()
result = svm.train_test()
# result = pd.read_excel(os.path.join('../../result/prediction/', 'USDJPY5_.csv'))

result = svm.add_loss_gain(result)
score = svm.get_step_score(result)

result['score'] = np.array(score)
print(result)
print(score)
io.to_excel(result)
