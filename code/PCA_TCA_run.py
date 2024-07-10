import pandas as pd
import numpy as np
import utils
# import model
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")
import scipy.stats as stats
import matplotlib.pyplot as plt


# data path
file_path = '../data/cumulative_2022_v3_9_domain.csv'

data = pd.read_csv(file_path)

column_to_variable_dict = np.load('../data/column_to_variable_dict.npy', allow_pickle=True).item()
variable_to_column_dict = np.load('../data/variable_to_column_dict.npy', allow_pickle=True).item()

value_label_dict = np.load('../data/value_labels.npy', allow_pickle=True).item()

target_variable = 'Voted_D_R'

'''Voted_D_R  {0.0: '0. Did not vote; DK/NA if voted; refused to say if', 1.0: '1. Democrat', 2.0: '2. Republican'}'''




data_train = data[(data['Voted_D_R'] == 1) | (data['Voted_D_R'] == 2)]
data_test = data[data['Voted'] == 1]

data_new = pd.concat([data_train, data_test])

missing_value = utils.missing_value_analysis(data_new)


threshold_list = [0.2, 0.3, 0.4, 0.5]


# must_include_list = ['urbanism']
must_include_list = None


folder_name = '../data/universal_predict/'

used_features, not_used_features, folder_name = utils.feature_filter(data_new, threshold_list,column_to_variable_dict, folder_name, must_include_list)

target_variable_list = ['Voted','Registered_voted','Voted_party','Vote_Nonvote_Pres','Voted_D_R']

race_variable_list = ['Race3','Race4','Race7']

religion_variable_list = ['religion']

index_variable_list = ['Year', ]

not_used_features = ['Pre_election_inten_vote']
# not_used_features = []


state_variable_list = ['State']

non_feature_list = target_variable_list +  race_variable_list + religion_variable_list + index_variable_list + not_used_features + state_variable_list

year_threshold = 1982



# only use samples in WA state

# data_train = data_train[data_train['State'] == 'WA'].reset_index(drop=True)
# data_test = data_test[data_test['State'] == 'WA'].reset_index(drop=True)


folder_name = folder_name + '/'+ str(year_threshold)+ '/'

# folder_name = folder_name + '/WA/'+ str(year_threshold)+ '/'


# filter out the samples whose year > year_threshold
data_train = data_train[data_train['Year'] > year_threshold].reset_index(drop=True)
data_test = data_test[data_test['Year'] > year_threshold].reset_index(drop=True)

data_new = pd.concat([data_train, data_test]).reset_index(drop=True)

print(data_train.shape)
print(data_test.shape)
print(data_new.shape)


numerical_feature_list, categorical_feature_list = utils.feature_type_analysis(data_new, used_features, non_feature_list)

target_variable = 'Voted_D_R'

Y_target_predict = utils.universal_predict(data_train,data_test, numerical_feature_list, categorical_feature_list, target_variable, value_label_dict, folder_name, group='', group_cat='')

Y_target_predict_TCA, Xs_new, Xt_new,X_continuous_categorical_train, X_continuous_categorical_test,D_R_stats = utils.universal_predict_TCA(data_train, data_test, numerical_feature_list, categorical_feature_list, target_variable, value_label_dict, folder_name, group='TCA', group_cat='')

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

Xs_new_pca = pca.fit_transform(Xs_new)

Xt_new_pca = pca.transform(Xt_new)

Xs_raw_pca = pca.fit_transform(X_continuous_categorical_train)

Xt_raw_pca = pca.transform(X_continuous_categorical_test)

index_vote_D = data_train[data_train['Voted_D_R'] == 1].index

index_vote_R = data_train[data_train['Voted_D_R'] == 2].index

scatter_size = 7

plt.scatter(Xs_new_pca[index_vote_D,0], Xs_new_pca[index_vote_D,1], c='b', label='voter-D', marker='o', alpha=0.5, s=scatter_size)
plt.scatter(Xs_new_pca[index_vote_R,0], Xs_new_pca[index_vote_R,1], c='r', label='voter-R', marker='x', alpha=0.5, s=scatter_size)

plt.scatter(Xt_new_pca[:,0], Xt_new_pca[:,1], c='g', label='non-voter', marker='^', alpha=0.5, s=scatter_size)

# plot the prediction difference samples
# plt.scatter(Xt_new_pca[differ_index,0], Xt_new_pca[differ_index,1], c='y', label='diff', marker='*', alpha=0.5, s=1)



plt.title('PCA on the first two components-TCA feature')
plt.legend()

# save the figure
# plt.savefig(folder_name + 'PCA_TCA_diff.png')
plt.savefig(folder_name + 'PCA_TCA-new.png')

plt.scatter(Xs_raw_pca[index_vote_D,0], Xs_raw_pca[index_vote_D,1], c='b', label='voter-D', marker='o', alpha=0.5, s=scatter_size)
plt.scatter(Xs_raw_pca[index_vote_R,0], Xs_raw_pca[index_vote_R,1], c='r', label='voter-R', marker='x', alpha=0.5, s=scatter_size)

plt.scatter(Xt_raw_pca[:,0], Xt_raw_pca[:,1], c='g', label='non-voter', marker='^', alpha=0.3, s=scatter_size)

# plt.scatter(Xt_raw_pca[differ_index,0], Xt_raw_pca[differ_index,1], c='y', label='diff', marker='*', alpha=0.5, s=1)

plt.title('PCA on the first two components-raw feature')
plt.legend()


# plt.savefig(folder_name + 'PCA_raw_diff.png')
plt.savefig(folder_name + 'PCA_raw-new.png')