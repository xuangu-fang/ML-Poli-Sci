
import pandas as  pd
import os

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


def feature_filter(data, missing_ratio_thr,column_to_variable_dict):
    """filter out the features based on the given thresholds of missing ratios and years"""

    missing_value = missing_value_analysis(data)

    # column filter-out: based on missing_ratio_thr: remove the features with missing value ratio > missing_ratio_thr

    # 确定每一行是否满足所有阈值条件
    all_conditions = True
    any_condition_not_met = False
    
    folder_name = '../data/'
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
    with open(folder_name + '/used_features.txt', 'w') as f:
        for item in used_features:
            f.write("%s (%s)\n" % (item, column_to_variable_dict[item]))

    # save the not used features names (row names)
    not_used_features = missing_value_not_used.index.tolist()
    with open(folder_name + '/not_used_features.txt', 'w') as f:
        for item in not_used_features:
            f.write("%s (%s)\n" % (item, column_to_variable_dict[item]))

    return used_features, not_used_features, folder_name
    





