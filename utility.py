import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def read_data(file_name):
    ''' 
    This subfunction reads the training data from the original excel file, and transform it into a dataframe.
    Input:
    * file_name: A string of the filename.
    Output:
    * df_data: A dataframe containing the data.
    '''
    tmp_df_ok_100 = pd.read_excel(file_name, sheet_name=0, usecols='C:N', header=3, nrows=30)
    tmp_df_ok_100['Label'] = 0
    tmp_df_ok_100['Speed'] = 100

    tmp_df_ok_500 = pd.read_excel(file_name, sheet_name=0, usecols='C:N', header=38, nrows=30)
    tmp_df_ok_500['Label'] = 0
    tmp_df_ok_500['Speed'] = 500

    tmp_df_ok_1000 = pd.read_excel(file_name, sheet_name=0, usecols='C:N', header=72, nrows=30)
    tmp_df_ok_1000['Label'] = 0
    tmp_df_ok_1000['Speed'] = 1000

    tmp_df_ko_100 = pd.read_excel(file_name, sheet_name=1, usecols='C:N', header=3, nrows=30)
    tmp_df_ko_100['Label'] = 1
    tmp_df_ko_100['Speed'] = 100

    tmp_df_ko_500 = pd.read_excel(file_name, sheet_name=1, usecols='C:N', header=37, nrows=30)
    tmp_df_ko_500['Label'] = 1
    tmp_df_ko_500['Speed'] = 500

    tmp_df_ko_1000 = pd.read_excel(file_name, sheet_name=1, usecols='C:N', header=80, nrows=30)
    tmp_df_ko_1000['Label'] = 1
    tmp_df_ko_1000['Speed'] = 1000

    df_data_org = pd.concat([tmp_df_ok_100, tmp_df_ok_500, tmp_df_ok_1000, tmp_df_ko_100, tmp_df_ko_500, tmp_df_ko_1000], ignore_index=True)

    return df_data_org


def data_visualization_2d_pca(x_clean_fill_nan, y, feature_list):
    ''' This function performs 2-d pca and visualize the result. '''
    # Use PCA to reduce the dimension to 2-D.
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_clean_fill_nan.loc[:, feature_list])
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, y], axis = 1)

    print(finalDf.head())
    print(pca.explained_variance_ratio_)

    # Visualize the data.
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [False, True]
    colors = ['r', 'g']
    marker = ['*', 'o']
    alpha = [.3, .3]
    for target, color, marker, alpha in zip(targets, colors, marker, alpha):
        indicesToKeep = finalDf['Label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50, marker=marker, alpha=alpha)
    ax.legend(targets)
    ax.grid()