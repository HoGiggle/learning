# coding: utf8
import datetime
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb


def base_process(data):
    user_int_fea = ['用户年龄', '用户网龄（月）', '当月通话交往圈人数', '近三个月月均商场出现次数', '当月网购类应用使用次数',
                    '当月物流快递类应用使用次数', '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数', '当月飞机类应用使用次数',
                    '当月火车类应用使用次数', '当月旅游资讯类应用使用次数']

    call_cost_fea = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）', '用户账单当月总费用（元）', '用户当月账户余额（元）']

    user_big_int_fea = ['当月网购类应用使用次数', '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数']

    # 异常值截断
    for col in user_int_fea + call_cost_fea:
        high = np.percentile(data[col].values, 99.9)
        low = np.percentile(data[col].values, 0.1)
        data.loc[data[col] > high, col] = high
        data.loc[data[col] < low, col] = low

    # 过大量级值取log平滑
    for col in user_big_int_fea:
        data[col] = data[col].map(lambda x: np.log1p(x))

    # 交通APP特征汇总
    data['交通APP次数'] = data['当月火车类应用使用次数'] + data['当月飞机类应用使用次数']
    # data = data.drop(columns=['当月火车类应用使用次数', '当月飞机类应用使用次数'])
    data = data.drop(columns=['用户编码'])

    return data


def xgb_train(feature_columns, x_train, y_train, x_test, y_test, modelprefix, num_round,
          print_importance_flag=False, save_model_flag=True):
    # print(np.shape(x_train),np.shape(y_train))
    # print(np.shape(x_test),np.shape(y_test))

    # 参数调优
    # x_train = np.concatenate((x_train, x_test))
    # y_train = np.concatenate((y_train, y_test))
    # other_params = {'learning_rate': 0.003, 'n_estimators': num_round, 'max_depth': 6, 'min_child_weight': 10, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.5, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 5, 'n_jobs':20}
    # model = xgb.XGBRegressor(**other_params)

    # optimized_GBM = GridSearchCV(estimator=model, param_grid={'min_child_weight': [10, 30, 50],
    # 'subsample': [0.5,0.6,0.7,0.8], 'reg_lambda': [5], 'learning_rate': [0.003],
    # 'n_estimators': [5000, 6000, 8000, 9000, 10000]}, scoring='neg_mean_absolute_error', cv=2, verbose=1, n_jobs=2)
    # optimized_GBM.fit(x_train, y_train)

    # print('每轮迭代运行结果:')
    # for indx, one in enumerate(optimized_GBM.cv_results_['params']):
    #     print(one, optimized_GBM.cv_results_['mean_test_score'][indx], optimized_GBM.cv_results_['std_test_score'][indx])
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    # return

    # 原始模型
    # other_params = {'learning_rate': 0.1, 'n_estimators': 400, 'max_depth': 7, 'min_child_weight': 10, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'n_jobs':10}
    # model = xgb.XGBRegressor(**other_params)
    # model.fit(x_train, y_train, eval_metric=metrics.mean_absolute_error)
    # # model.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric=metrics.mean_absolute_error)
    # y_pred = model.predict(x_test)
    # print('[%s]test mae: %.4f'%(datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S'), metrics.mean_absolute_error(y_test,y_pred)))

    other_params = {'learning_rate': 0.003, 'n_estimators': 8000, 'max_depth': 6, 'min_child_weight': 10, 'seed': 0,
                    'subsample': 0.6, 'colsample_bytree': 0.5, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 5, 'n_jobs':20}
    model = xgb.XGBRegressor(**other_params)
    model.fit()
    model.fit(x_train, y_train, eval_metric=mean_absolute_error)

    # model.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric=metrics.mean_absolute_error)
    # y_pred = model.predict(x_test)
    # mae = metrics.mean_absolute_error(y_test,y_pred)
    # print('[%s]test mae: %.4f'%(datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S'), mae))
    # print('[%s]test score: %.4f'%(datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S'), 1/(1+mae)))

    if print_importance_flag:
        print("[%s]importance:" % (datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')))
        importance = list(zip(list(feature_columns), model.feature_importances_))
        importance.sort(key= lambda k:k[1], reverse=True)
        for one in importance:
            print("\t%s: %.6f" % (one[0], one[1]))

    if save_model_flag:
        joblib.dump(model, modelprefix + "." + str(num_round)+'.model')
        # model.save_model(modelprefix+"."+str(num_round)+'.model')
    return model



if __name__ == '__main__':
    train_df = pd.read_csv('/Users/giggle/Downloads/train_dataset.csv')
    test_df = pd.read_csv('/Users/giggle/Downloads/test_dataset.csv')

    # 特征处理
    train_feature = base_process(train_df.drop(columns=['信用分']))
    test_feature = base_process(test_df)

    # train data generate
    train_label = train_df['信用分']
    x_train, x_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.2, random_state=10)


    # model train
    # Ridge
    # model = linear_model.Ridge(alpha=.5)
    # model.fit(x_train, y_train)

    # GBRT
    # ********** cross validation
    # param_test1 = {'min_samples_split': range(205, 240, 5)}
    # model = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1, n_estimators=70,
    #                                                          max_depth=7, random_state=0),
    #                      param_grid=param_test1, scoring='neg_mean_squared_error', iid=False, cv=3)
    # model.fit(x_train, y_train)
    # print model.cv_results_, model.best_params_, model.best_score_

    # params = {'n_estimators': 650, 'subsample': 0.9, 'learning_rate': 0.03, 'loss': 'lad', 'random_state': 0,
    #           'max_depth': 7, 'min_samples_split': 1060, 'min_samples_leaf': 120, 'max_features': 19}
    # model = GradientBoostingRegressor(**params)
    # model.fit(x_train, y_train)
    x_train, x_val, y_train, y_val = train_test_split(train_feature, train_label, test_size=0.2, random_state=10)

    params = {'learning_rate': 0.003, 'n_estimators': 8000, 'max_depth': 6, 'min_child_weight': 10, 'seed': 0,
          'subsample': 0.7, 'colsample_bytree': 0.5, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 5, 'n_jobs':20}
    model = xgb.XGBRegressor(**params)
    # model.fit(x_train, x_train, eval_metric=mean_absolute_error)

    # valdata = xgb.DMatrix(x_val.values, label=y_val.values)

    model.fit(x_train, y_train, early_stopping_rounds=10, eval_metric=mean_absolute_error, eval_set=[(x_val, y_val)])

    # predict
    # mse = mean_squared_error(y_train, model.predict(x_train))
    # print 'train mse = ', mse
    # mse = mean_squared_error(y_test, model.predict(x_test))
    # print 'test mse = ', mse
    print 'train score = ', (1 / (mean_absolute_error(y_train, model.predict(x_train)) + 1))
    print 'train score = ', (1 / (mean_absolute_error(y_test, model.predict(x_test)) + 1))


    # test data
    # test_pred = model.predict(test_feature)
    # res = pd.concat([test_df['用户编码'], pd.DataFrame(map(int, test_pred))], axis=1)
    # res.columns = ['id', 'score']
    # res.to_csv(r'/Users/giggle/Downloads/result_gbrt_0.03_650_7_1060_120_19.csv', columns=['id', 'score'], index=False, sep=',')

