import numpy as np
import pandas as pd

train = pd.read_csv('../../data/train.csv')
submission = pd.read_csv('../../data/sample_submission.csv')

mean_std = 0.94*train.mean(axis=0) - 0.12*train.std(axis=0)
print train.mean(axis=0)
print(mean_std)

mean_std['adult_males'] = 5
mean_std['subadult_males'] = 4
mean_std['adult_females'] = 26
mean_std['juveniles'] = 15
mean_std['pups'] = 11

for c in submission.columns:
    if c != 'test_id':
        submission[c] = int(mean_std[c])
#submission.to_csv('submission.csv', index=False)
print(submission.head(10))
