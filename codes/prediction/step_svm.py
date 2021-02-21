import codes.prediction.io as io
import codes.prediction.svm_predictor as svm_predictor
import pandas as pd
import os

svm = svm_predictor.SVM()
result = svm.step_train_test()

# result = pd.read_excel(os.path.join('../../result/prediction/', 'USDJPY5_6000.xls'))
result = svm.add_loss_gain(result)
score = svm.get_step_score(result)

print(result)
print(score)
io.to_excel(result)
