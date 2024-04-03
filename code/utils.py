
import pandas as  pd
import os
import numpy as np
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from transfer_learn import TCA

def missing_value_analysis(data):

    """ 
    analysis the missing value of each column in differernt period:
    # 1. overall missing value and percentage
    # 2. missing value and percentage in recent 10 years ("Year" >= 2012)")
    # 3. missing value and percentage in recent 20 years ("Year" >= 2002)")
    # 4. missing value and percentage in recent 30 years ("Year" >= 1992)")
    # 5. missing value and percentage in recent 40 years ("Year" >= 1982)")
    # 6. missing value and percentage in recent 50 years ("Year" >= 1972)")
    # 7. missing value and percentage in recent 60 years ("Year" >= 1962)")
    save the result in csv file, the first column is the feature name
    """


    # get the number of missing value of each column
    missing_value_num = data.isnull().sum()
    # get the percentage of missing value of each column
    missing_value_percentage = missing_value_num / len(data)

    missing_value_percentage_10 = data[data['Year'] >= 2012].isnull().sum() / len(data[data['Year'] >= 2012])
    missing_value_percentage_20 = data[data['Year'] >= 2002].isnull().sum() / len(data[data['Year'] >= 2002])
    missing_value_percentage_30 = data[data['Year'] >= 1992].isnull().sum() / len(data[data['Year'] >= 1992])
    missing_value_percentage_40 = data[data['Year'] >= 1982].isnull().sum() / len(data[data['Year'] >= 1982])
    missing_value_percentage_50 = data[data['Year'] >= 1972].isnull().sum() / len(data[data['Year'] >= 1972])
    missing_value_percentage_60 = data[data['Year'] >= 1962].isnull().sum() / len(data[data['Year'] >= 1962])
    missing_value_percentage_70 = data[data['Year'] >= 1952].isnull().sum() / len(data[data['Year'] >= 1952])

    # get the variable name of each column by using the column_to_variable_dict
    # missing_value_num.index = column_to_variable_dict['variable']


    # combine the result
    missing_value = pd.concat([missing_value_num, missing_value_percentage,
                               missing_value_percentage_10, missing_value_percentage_20,
                               missing_value_percentage_30, missing_value_percentage_40,
                               missing_value_percentage_50, missing_value_percentage_60,    missing_value_percentage_70], axis=1)
    missing_value.columns = ['missing_value_num', 'missing_value_percentage',
                                'missing_value_percentage_10(>=2012)', 'missing_value_percentage_20(>=2002)',
                                'missing_value_percentage_30(>=1992)', 'missing_value_percentage_40(>=1982)',
                                'missing_value_percentage_50(>=1972)', 'missing_value_percentage_60(>=1962)', 'missing_value_percentage_60(>=1952)']

    # sort the result by missing value percentage
    missing_value = missing_value.sort_values(by='missing_value_percentage', ascending=False)


    return missing_value


def feature_filter(data, missing_ratio_thr,column_to_variable_dict, folder_name = '../data/',must_include_list=None):
    """filter out the features based on the given thresholds of missing ratios and years"""

    missing_value = missing_value_analysis(data)

    # column filter-out: based on missing_ratio_thr: remove the features with missing value ratio > missing_ratio_thr

    # 确定每一行是否满足所有阈值条件
    all_conditions = True
    any_condition_not_met = False
    
    
    for i, threshold in enumerate(missing_ratio_thr):
        column_name = missing_value.columns[i + 2]  # 第n个阈值对应第n+2列
        current_condition = missing_value[column_name] < threshold
        all_conditions = all_conditions & current_condition
        any_condition_not_met = any_condition_not_met | ~current_condition

        folder_name = folder_name +'_threshold_%s'%(str(10*(i+1))) + '_' + str(threshold) 
    
    # 应用筛选条件
    missing_value_used = missing_value[all_conditions]
    missing_value_not_used = missing_value[any_condition_not_met]

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # save the used features names (row names) and the variable names
    used_features = missing_value_used.index.tolist()

    not_used_features = missing_value_not_used.index.tolist()

    if must_include_list is not None:
        for feature in must_include_list:
            if feature not in used_features:
                used_features.append(feature)
                not_used_features.remove(feature)
    
    with open(folder_name + '/used_features.txt', 'w') as f:
        for item in used_features:
            f.write("%s (%s)\n" % (item, column_to_variable_dict[item]))

    # save the not used features names (row names)
    
    with open(folder_name + '/not_used_features.txt', 'w') as f:
        for item in not_used_features:
            f.write("%s (%s)\n" % (item, column_to_variable_dict[item]))

    return used_features, not_used_features, folder_name


def feature_type_analysis(data_new, used_features, non_feature_list):
    """
    extract the numerical and categorical features from the used features
    """

# go through all used features, check the num of the categories of each feature: if the num of categories > 10, then this feature is a continuous/numerical feature, otherwise, this feature is a categorical feature-> need to do one-hot encoding

    numerical_feature_list = []
    categorical_feature_list = []

    for feature in used_features:

        if feature not in non_feature_list:

            if len(data_new[feature].value_counts()) > 10:
                numerical_feature_list.append(feature)
            else:
                categorical_feature_list.append(feature)

    print('number of numerical features: ', len(numerical_feature_list))

    print('number of categorical features: ', len(categorical_feature_list))

    print('numerical features list:',numerical_feature_list)

    return numerical_feature_list, categorical_feature_list


def group_split_religon(data_new):
    '''group split by religon'''

    # religion  {0.0: '0. DK; NA; refused to answer; no Pre IW; no Post IW;', 1.0: '1. Protestant', 2.0: '2. Catholic [Roman Catholic]', 3.0: '3. Jewish', 4.0: '4. Other and none (also includes DK preference)'}

    # print(data_new['religion'].value_counts())

    data_religion_dict={}

    data_religion_dict['Protestant'] = data_new[data_new['religion'] == 1]
    data_religion_dict['Catholic']  = data_new[data_new['religion'] == 2]
    data_religion_dict['Jewish']      = data_new[data_new['religion'] == 3]
    data_religion_dict['Other']  = data_new[(data_new['religion'] == 4 ) | (data_new['religion'] == 0)]

    print('number of samples of Protestant: ', len(data_religion_dict['Protestant']))
    print('number of samples of Catholic: ', len(data_religion_dict['Catholic']))
    print('number of samples of Jewish: ', len(data_religion_dict['Jewish'] ))
    print('number of samples of Other: ', len(data_religion_dict['Other'] ))
        

    return data_religion_dict

def group_split_race7(data_new):

    '''group split by race

    Race7  
    {1.0: '1. White non-Hispanic (1948-2012)', 
    2.0: '2. Black non-Hispanic (1948-2012)', 
    3.0: '3. Asian or Pacific Islander, non-Hispanic (1966-2012)', 4.0: 
    '4. American Indian or Alaska Native non-Hispanic (1966-2012)',
    5.0: '5. Hispanic (1966-2012)', 
    6.0: '6. Other or multiple races, non-Hispanic (1968-2012)',
    7.0: '7. Non-white and non-black (1948-1964)', 9.0: '9. Missing'}

    '''
    # print(data_new['Race3'].value_counts())

    # print(data_new['Race7'].value_counts())

    data_race7_dict={}
    data_race7_dict['White'] = data_new[data_new['Race7'] == 1]
    data_race7_dict['Black'] = data_new[data_new['Race7'] == 2]
    data_race7_dict['Asian'] = data_new[data_new['Race7'] == 3]
    data_race7_dict['American_Indian'] = data_new[data_new['Race7'] == 4]
    data_race7_dict['Hispanic'] = data_new[data_new['Race7'] == 5]
    data_race7_dict['Other'] = data_new[(data_new['Race7'] == 6) | (data_new['Race7'] == 7) | (data_new['Race7'] == 9)]

    print('number of samples of White: ', len(data_race7_dict['White']))
    print('number of samples of Black: ', len(data_race7_dict['Black']))
    print('number of samples of Asian: ', len(data_race7_dict['Asian']))
    print('number of samples of American_Indian: ', len(data_race7_dict['American_Indian']))
    print('number of samples of Hispanic: ', len(data_race7_dict['Hispanic']))
    print('number of samples of Other: ', len(data_race7_dict['Other']))

    return data_race7_dict



# def get_feature_name_category_name(string, enc, value_label_dict):
#     feature_id = int(string.split('_')[0][1:])
#     category_index = int(float(string.split('_')[1]))

#     feature_name = enc.feature_names_in_[feature_id]
    

#     if category_index == -1:
#         category_name = 'Missing'
#     else:
#         category_name = value_label_dict[feature_name][category_index]

#     return feature_name, category_name

# def enc_feature_list(initial_list, enc, value_label_dict):
#     new_list = []

#     for string in initial_list:
#         feature_name, category_name = get_feature_name_category_name(string, enc, value_label_dict)
#         new_list.append((feature_name+'_'+ category_name))
#     return new_list
def custom_combiner(feature, category):
    return str(feature) + "_XX_" + str(category)

def get_feature_name_category_name(string, enc, value_label_dict):
    # feature_id = int(string.split('_XX_')[0][1:])
    feature_name = string.split('_XX_')[0]
    category_index = float(string.split('_XX_')[1])
    
    if category_index == -1:
        category_name = 'Missing'
    elif category_index not in value_label_dict[feature_name].keys():
        category_name = 'unmatched category'
    else: 
        category_name = value_label_dict[feature_name][category_index]

    return feature_name, category_name


def enc_feature_list(initial_list, enc, value_label_dict):
    new_list = []

    for string in initial_list:
        feature_name, category_name = get_feature_name_category_name(string, enc, value_label_dict)
        new_list.append((feature_name+'_'+ category_name))
    return new_list    


def feature_process(data, numerical_feature_list, categorical_feature_list, target_variable,value_label_dict):
    data_XY = data[numerical_feature_list + categorical_feature_list+[target_variable]]
    # data_XY = data_XY[data_XY.notnull().all(axis=1)]
    data_XY = data_XY.reset_index(drop=True)
    print(data_XY.shape)

    X_continuous = data_XY[numerical_feature_list]
    X_categorical = data_XY[categorical_feature_list]
    Y_target = data_XY[target_variable]

    # non-voters are labeled as 1, voters are labeled as 0
    # Democratic are labeled as 1, Repub are labeled as 0

    Y_target = Y_target.apply(lambda x: 1 if x == 1 else 0)

    # impute + process(one-hot)  categorical features (also get the new names)

    X_categorical_imp = X_categorical.fillna(-1)

    enc = OneHotEncoder(handle_unknown='ignore',feature_name_combiner=custom_combiner)

    enc.fit(X_categorical_imp)

    X_categorical_transformed = enc.transform(X_categorical_imp).toarray()

    initial_list = enc.get_feature_names_out().tolist()
    enc_categorical_feature_list = utils.enc_feature_list(initial_list, enc, value_label_dict)    

    #impute + process(normalize) the numerical features
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    X_continuous_imp = imp.fit_transform(X_continuous)

    X_continuous_transformed = StandardScaler().fit_transform(X_continuous_imp)


    return X_categorical_transformed, X_continuous_transformed, Y_target, enc_categorical_feature_list



def cross_validation(X, Y, model, k = 5):
    
    kf = KFold(n_splits=k, random_state=1, shuffle=True)
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    roc_auc_list = []
    importance_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_list.append(accuracy_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        roc_auc_list.append(roc_auc_score(y_test, y_pred))
        importance_list.append(model.coef_[0])

    return accuracy_list, recall_list, precision_list, f1_list, roc_auc_list, importance_list


def universal_predict(X_train, Y_train, X_test, model):

    # apply
    
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    return y_pred

from imblearn.over_sampling import RandomOverSampler


def cross_validation_imb(X, Y, model, k = 5):
    
    kf = KFold(n_splits=k, random_state=1, shuffle=True)
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    roc_auc_list = []
    importance_list = []

    pipeline = make_pipeline(SMOTE(random_state=42), model)


    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        pipeline.fit(X_train, y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_list.append(accuracy_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        roc_auc_list.append(roc_auc_score(y_test, y_pred))
        # importance_list.append(model.coef_[0])

        importance_list.append(pipeline[-1].coef_[0])

    return accuracy_list, recall_list, precision_list, f1_list, roc_auc_list, importance_list

def cross_validation_imb_downsampling(X, Y, model, k = 5):
    
    kf = KFold(n_splits=k, random_state=1, shuffle=True)
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    roc_auc_list = []
    importance_list = []

    pipeline = make_pipeline(RandomUnderSampler(sampling_strategy=0.5,random_state=42), model)


    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        pipeline.fit(X_train, y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_list.append(accuracy_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        roc_auc_list.append(roc_auc_score(y_test, y_pred))
        # importance_list.append(model.coef_[0])

        importance_list.append(pipeline[-1].coef_[0])

    return accuracy_list, recall_list, precision_list, f1_list, roc_auc_list, importance_list

def feature_importance_analysis(data_group, numerical_feature_list, categorical_feature_list, target_variable, value_label_dict, folder_name, group='', group_cat=''):
     
    X_categorical_transformed, X_continuous_transformed, Y_target, enc_categorical_feature_list = utils.feature_process(data_group, numerical_feature_list, categorical_feature_list, target_variable,value_label_dict)

    X_continuous_categorical = np.concatenate((X_continuous_transformed, X_categorical_transformed), axis=1)

    model = LogisticRegression(l1_ratio = 0.5, max_iter = 500, solver = 'saga', penalty = 'elasticnet')

    accuracy_list, recall_list, precision_list, f1_list, roc_auc_list, importance_list = utils.cross_validation(X_continuous_categorical, Y_target, model, k = 5)


    # use imbalanced learn to deal with the imbalanced data
    # accuracy_list, recall_list, precision_list, f1_list, roc_auc_list, importance_list = utils.cross_validation_imb(X_continuous_categorical, Y_target, model, k = 5)


    print('average accuracy: ', np.mean(accuracy_list))
    print('average recall: ', np.mean(recall_list))
    print('average precision: ', np.mean(precision_list))
    print('average f1 score: ', np.mean(f1_list))
    print('average roc auc score: ', np.mean(roc_auc_list))

    # build the feature importance dataframe
    feature_importance = pd.DataFrame({'feature': numerical_feature_list + enc_categorical_feature_list, 'importance': np.mean(importance_list, axis=0)})

     # further process the feature importance dataframe, drop the features whose name includes {DK', 'NA', 'RF', 'Missing'}
    feature_importance_effect = feature_importance[~feature_importance['feature'].str.contains('DK|NA|RF|Missing')]


    top_15_positive = feature_importance_effect.sort_values('importance', ascending = False).head(15)
    top_15_negative = feature_importance_effect.sort_values('importance', ascending = True).head(15)


    # build a folder to save the results

    if group_cat == '':
        sub_folder_name = folder_name + group + '/'
    else:
        sub_folder_name = folder_name + group + '/' + group_cat + '/' 
    
    if not os.path.exists(sub_folder_name):
        os.makedirs(sub_folder_name)

    # recall: the non-voter are the positive samples, the voter are the negative samples

    feature_importance.to_csv(sub_folder_name + 'feature_importance_full.csv', index = False)
    feature_importance_effect.to_csv(sub_folder_name + 'feature_importance_effect.csv', index = False)
    top_15_positive.to_csv(sub_folder_name + 'top_15_non_voter.csv', index = False)
    top_15_negative.to_csv(sub_folder_name + 'top_15_voter.csv', index = False)

    # add the ratio of the positive samples(non-voter) in the group
    non_voter_ratio = len(data_group[data_group[target_variable] == 1]) / len(data_group)

    # save the mean of the metrics
    metrics = pd.DataFrame({ 'non-voter-ratio': non_voter_ratio ,   'accuracy': np.mean(accuracy_list), 'recall': np.mean(recall_list), 'precision': np.mean(precision_list), 'f1': np.mean(f1_list), 'roc_auc': np.mean(roc_auc_list)}, index = [0])
    metrics.to_csv(sub_folder_name + 'metrics.csv', index = False)



def universal_predict(data_source,data_target, numerical_feature_list, categorical_feature_list, target_variable, value_label_dict, folder_name, group='', group_cat=''):
     
    N1 = len(data_source)
    N2 = len(data_target)

    data_group = pd.concat([data_source, data_target]).reset_index(drop=True)

    X_categorical_transformed, X_continuous_transformed, Y_target, enc_categorical_feature_list = utils.feature_process(data_group, numerical_feature_list, categorical_feature_list, target_variable,value_label_dict)

    X_continuous_categorical = np.concatenate((X_continuous_transformed, X_categorical_transformed), axis=1)

    # only use the source data to train the model
    Y_target_train = Y_target[:N1]

    X_continuous_categorical_train = X_continuous_categorical[:N1]

    X_continuous_categorical_test = X_continuous_categorical[N1:]



    model = LogisticRegression(l1_ratio = 0.5, max_iter = 500, solver = 'saga', penalty = 'elasticnet')

    accuracy_list, recall_list, precision_list, f1_list, roc_auc_list, importance_list = utils.cross_validation(X_continuous_categorical_train, Y_target_train, model, k = 5)

    # use imbalanced learn to deal with the imbalanced data
    # accuracy_list, recall_list, precision_list, f1_list, roc_auc_list, importance_list = utils.cross_validation_imb(X_continuous_categorical, Y_target, model, k = 5)


    print('average accuracy: ', np.mean(accuracy_list))
    print('average recall: ', np.mean(recall_list))
    print('average precision: ', np.mean(precision_list))
    print('average f1 score: ', np.mean(f1_list))
    print('average roc auc score: ', np.mean(roc_auc_list))

    # build the feature importance dataframe
    feature_importance = pd.DataFrame({'feature': numerical_feature_list + enc_categorical_feature_list, 'importance': np.mean(importance_list, axis=0)})

     # further process the feature importance dataframe, drop the features whose name includes {DK', 'NA', 'RF', 'Missing'}
    feature_importance_effect = feature_importance[~feature_importance['feature'].str.contains('DK|NA|RF|Missing')]


    top_15_positive = feature_importance_effect.sort_values('importance', ascending = False).head(15)
    top_15_negative = feature_importance_effect.sort_values('importance', ascending = True).head(15)

    # build a folder to save the results

    if group_cat == '':
        sub_folder_name = folder_name + group + '/'
    else:
        sub_folder_name = folder_name + group + '/' + group_cat + '/' 
    
    if not os.path.exists(sub_folder_name):
        os.makedirs(sub_folder_name)

    # recall: the non-voter are the positive samples, the voter are the negative samples

    feature_importance.to_csv(sub_folder_name + 'feature_importance_full.csv', index = False)
    feature_importance_effect.to_csv(sub_folder_name + 'feature_importance_effect.csv', index = False)
    top_15_positive.to_csv(sub_folder_name + 'top_15_Demo.csv', index = False)
    top_15_negative.to_csv(sub_folder_name + 'top_15_Repub.csv', index = False)

    # add the ratio of the positive samples(non-voter) in the group
    non_voter_ratio = len(data_group[data_group[target_variable] == 1]) / len(data_group)

    # save the mean of the metrics
    metrics = pd.DataFrame({ 'non-voter-ratio': non_voter_ratio ,   'accuracy': np.mean(accuracy_list), 'recall': np.mean(recall_list), 'precision': np.mean(precision_list), 'f1': np.mean(f1_list), 'roc_auc': np.mean(roc_auc_list)}, index = [0])
    metrics.to_csv(sub_folder_name + 'metrics.csv', index = False)


    #  apply the universal model to the target data
    model.fit(X_continuous_categorical_train, Y_target_train)
    Y_target_predict = model.predict(X_continuous_categorical_test)

    # value counts of the prediction
    print(pd.Series(Y_target_predict).value_counts())

    # save the prediction results as a csv file
    data_target['prediction'] = Y_target_predict
    data_target.to_csv(sub_folder_name + 'prediction.csv', index = False)

    return Y_target_predict



def universal_predict_TCA(data_source,data_target, numerical_feature_list, categorical_feature_list, target_variable, value_label_dict, folder_name, group='', group_cat='',dim=100):

    # apply transfer component analysis (TCA) to the data

     
    N1 = len(data_source)
    N2 = len(data_target)

    data_group = pd.concat([data_source, data_target]).reset_index(drop=True)

    X_categorical_transformed, X_continuous_transformed, Y_target, enc_categorical_feature_list = utils.feature_process(data_group, numerical_feature_list, categorical_feature_list, target_variable,value_label_dict)

    X_continuous_categorical = np.concatenate((X_continuous_transformed, X_categorical_transformed), axis=1)

    # only use the source data to train the model
    Y_target_train = Y_target[:N1]

    X_continuous_categorical_train = X_continuous_categorical[:N1]

    X_continuous_categorical_test = X_continuous_categorical[N1:]

    # transfer learning step

    TCA_model = TCA(kernel_type='primal', dim=dim, lamb=1, gamma=1)
    Xs_new, Xt_new = TCA_model.fit(X_continuous_categorical_train, X_continuous_categorical_test)


    model = LogisticRegression(l1_ratio = 0.5, max_iter = 500, solver = 'saga', penalty = 'elasticnet')

    accuracy_list, recall_list, precision_list, f1_list, roc_auc_list, importance_list = utils.cross_validation(Xs_new, Y_target_train, model, k = 5)

    # use imbalanced learn to deal with the imbalanced data
    # accuracy_list, recall_list, precision_list, f1_list, roc_auc_list, importance_list = utils.cross_validation_imb(X_continuous_categorical, Y_target, model, k = 5)


    print('average accuracy: ', np.mean(accuracy_list))
    print('average recall: ', np.mean(recall_list))
    print('average precision: ', np.mean(precision_list))
    print('average f1 score: ', np.mean(f1_list))
    print('average roc auc score: ', np.mean(roc_auc_list))

    # build a folder to save the results

    if group_cat == '':
        sub_folder_name = folder_name + group + '/'
    else:
        sub_folder_name = folder_name + group + '/' + group_cat + '/' 
    
    if not os.path.exists(sub_folder_name):
        os.makedirs(sub_folder_name)

    # add the ratio of the positive samples(non-voter) in the group
    non_voter_ratio = len(data_group[data_group[target_variable] == 1]) / len(data_group)

    # save the mean of the metrics
    metrics = pd.DataFrame({ 'non-voter-ratio': non_voter_ratio ,   'accuracy': np.mean(accuracy_list), 'recall': np.mean(recall_list), 'precision': np.mean(precision_list), 'f1': np.mean(f1_list), 'roc_auc': np.mean(roc_auc_list)}, index = [0])
    metrics.to_csv(sub_folder_name + 'metrics.csv', index = False)


    #  apply the universal model to the target data
    model.fit(Xs_new, Y_target_train)
    Y_target_predict = model.predict(Xt_new)

    # value counts of the prediction
    print(pd.Series(Y_target_predict).value_counts())

    # save the prediction results as a csv file
    data_target['prediction'] = Y_target_predict
    data_target.to_csv(sub_folder_name + 'prediction.csv', index = False)

    return  Y_target_predict, Xs_new, Xt_new,X_continuous_categorical_train, X_continuous_categorical_test



