
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
    else:
        category_name = value_label_dict[feature_name][category_index]

    return feature_name, category_name


def enc_feature_list(initial_list, enc, value_label_dict):
    new_list = []

    for string in initial_list:
        feature_name, category_name = get_feature_name_category_name(string, enc, value_label_dict)
        new_list.append((feature_name+'_'+ category_name))
    return new_list    

def custom_combiner(feature, category):
    return str(feature) + "_" + str(category)

def feature_process(data, numerical_feature_list, categorical_feature_list, target_variable,value_label_dict):
    data_XY = data[numerical_feature_list + categorical_feature_list+[target_variable]]
    # data_XY = data_XY[data_XY.notnull().all(axis=1)]
    data_XY = data_XY.reset_index(drop=True)
    print(data_XY.shape)

    X_continuous = data_XY[numerical_feature_list]
    X_categorical = data_XY[categorical_feature_list]
    Y_target = data_XY[target_variable]

    # impute + process(one-hot)  categorical features (also get the new names)

    X_categorical_imp = X_categorical.fillna(-1)

    enc = OneHotEncoder(handle_unknown='ignore',feature_name_combiner=custom_combiner)

    enc.fit(X_categorical_imp)

    X_categorical_transformed = enc.transform(X_categorical_imp).toarray()

    initial_list = enc.get_feature_names().tolist()
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
