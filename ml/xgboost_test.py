import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt


# clf = xgb.Booster({'nthread':4}) #init model
# clf.load_model("/Users/giggle/Work/data/0100.model") # load data
#
#
# feat_imp = pd.Series(clf.get_fscore()).sort_values(ascending=False)
# print "all feature size:"
# print feat_imp.size
# print "head 10 feature:"
# # print "last 10 feature:"
# # print feat_imp.tail(10)
#
#
# # feat_imp.head(30).plot(kind='bar', title='Feature Importances')
# # plt.ylabel('Feature Importance Score')
# # plt.show()
df = pd.DataFrame({'a':np.random.randn(1000)+1,'b':np.random.randn(1000),'c':
    np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])

df.hist(bins=20)
plt.show()