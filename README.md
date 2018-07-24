# Credit_Score_Process
The whole process to establish a tree-based credit model in python.


# A SMALL DEMO 
import sys
sys.path.append("/home/mlp/notebooks/functions")
from function import * 
mt = model_train(numlst="./reports/less_1y/num_list_1.csv", charlst="./reports/less_1y/char_list_1.csv", tgt='dlqnt_30_flag')

params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 200,
      'nthread': 20, 'reg_alpha': 0.01, 'reg_lambda': 0.01
         }
xgb_classifier = PMMLPipeline(steps=[
                    ('scaler', RobustScaler(quantile_range=(5.0, 95.0))),
                    ('GBDT', xgb.XGBClassifier(**params))
                ])
                
xgb_classifier.fit(dev_x, dev_y)
mt.some_reports(dev, val, oft, oft2, oft3, xgb_classifier, 10, 20, 'GBDT', 0, 0, './reports/less_1y/test.xlsx')
