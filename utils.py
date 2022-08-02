import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, Normalizer, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy import stats


def plot_count(df, label, rotate=False, hue_flag=False):
    plt.figure(figsize=(6, 6), dpi=100)
    hue = None
    if hue_flag:
        hue = 'Stay'
    sns.countplot(x=label, data=df, order=df[label].value_counts().index, hue=hue)
    plt.title(f'{label}')
    if rotate:
        plt.xticks(rotation=90)


def plot_box(df, label):
    plt.figure(figsize=(5, 5), dpi=100)
    sns.boxplot(x=label, data=df)
    plt.title(f'{label}')
    if len(df[label]) > 5:
        plt.xticks(rotation=90)


def bivariate_plot(df, x, y):
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, y=f'{y}', x=f'{x}', x_jitter=0.3, scatter_kws={'alpha': 1 / 3})


def remove_duplicates(df):
    if len(df[df.duplicated()]) > 0:
        df.drop_duplicates(inplace=True)
    return


def find_missing(df):
    missing = (100 * df.isnull().sum() / len(df)).sort_values()
    missing = missing[missing > 0].sort_values()
    return missing


def handle_missing(df):
    missing = find_missing(df)
    if len(missing.axes[0]) > 0:
        df.dropna(axis=0, inplace=True)
    return df


def plot_missing(df):
    try:
        missing = find_missing(df)
        plt.figure(figsize=(4, 4), dpi=100)
        sns.barplot(x=missing.index, y=missing)
        plt.xticks(rotation=90)
    except ValueError:
        print('No Missing Values!')


def outliers(df, label):
    # q75, q25 = np.percentile(df[label], [75, 25])
    # iqr = q75 - q25
    # lower = q25 - 1.5 * iqr
    # upper = q75 + 1.5 * iqr
    # df_filtered = df[(df[label] < upper) & (df[label] > lower)]
    # df_filtered = df_filtered[np.abs(df_filtered[f'{label}'] - df_filtered[f'{label}'].mean()) <=
    # (3*df_filtered[f'{label}'].std())]
    # return df_filtered
    data = df[(np.abs(stats.zscore(df[f'{label}'])) < 1.5)]
    return data


def normalize(df):
    normalizer = Normalizer()
    numeric_cols = df.select_dtypes(include=np.number)
    correct_num = numeric_cols.drop(['Hospital_code', 'City_Code_Hospital', 'City_Code_Patient', 'Bed Grade'], axis=1)
    df['Admission_Deposit'] = df['Admission_Deposit'].apply(int)
    normalized_data = df.copy()
    features = normalized_data[correct_num.columns]
    normalizer.fit_transform(features.values)
    normalized_data[correct_num.columns] = features
    return normalized_data


def get_categories(df):
    categories = []
    categorical_features = df.select_dtypes(exclude=np.number)
    correct_features = df[['Hospital_code', 'City_Code_Hospital', 'City_Code_Patient', 'Bed Grade']
                          + categorical_features.columns.to_list()]
    for c in correct_features.columns:
        categories.append({'label': str(c), 'value': str(c)})
    return categories


def get_numerical(df):
    numeric_cols = df.select_dtypes(include=np.number)
    correct_num = numeric_cols.drop(['Hospital_code', 'City_Code_Hospital', 'City_Code_Patient', 'Bed Grade'], axis=1)
    if 'case_id' in correct_num.columns:
        correct_num.drop('case_id', axis=1, inplace=True)
    if 'patientid' in correct_num.columns:
        correct_num.drop('patientid', axis=1, inplace=True)
    numerical = []
    for c in correct_num.columns:
        df[c].apply(int)
        numerical.append({'label': str(c), 'value': str(c)})
    return numerical


def categorical_encoding(df):
    """
    A function to perform categorical encoding
    :param df: The dataframe
    :param encode: type of encode - either ordinal or label
    :param categorical_features: The features to be encoded
    :return: dataframe with encoded features and the features
    """
    categorical_features = df.select_dtypes(exclude=np.number)
    encoder = OrdinalEncoder()
    df[categorical_features.columns] = encoder.fit_transform(categorical_features)
    return df, categorical_features


def scale_data(df):
    """
    A function to scale the numerical features in the data
    :param df:
    :return:
    """
    columns_list = ['Type of Admission', 'Available Extra Rooms in Hospital', 'Visitors with Patient',
                    'Admission_Deposit']
    scaled_data = df.copy()
    scaler = StandardScaler()
    scaled_data[columns_list] = scaler.fit_transform(df[columns_list])
    return scaled_data


def clean_data_pipeline(data, explore=True):
    data.dropna(axis=0, inplace=True)
    if 'case_id' in data.columns:
        data.drop('case_id', axis=1, inplace=True)
    if 'patientid' in data.columns:
        data.drop('patientid', axis=1, inplace=True)

    numeric_cols = data.select_dtypes(include=np.number)
    correct_num = numeric_cols.drop(['Hospital_code', 'City_Code_Hospital', 'City_Code_Patient', 'Bed Grade'], axis=1)

    for c in correct_num.columns:
        data = outliers(data, c)
    if not explore:
        data = scale_data(data)
        data, _ = categorical_encoding(data)
    return data


def oversample(x, y, method):
    if method == "smote":
        model = SMOTE()
        X, Y = model.fit_resample(x, y)
        return X, Y
