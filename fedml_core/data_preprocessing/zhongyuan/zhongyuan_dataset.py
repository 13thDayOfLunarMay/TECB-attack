import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re

from .zhongyuan_feature_group import all_feature_list, qualification_feat, \
    loan_feat, debt_feat, repayment_feat
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import random

work_year_map = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10
}

class_map = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7
}


def normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_scaled = (x - mu) / sigma
    '''
    mu_dir = data_dir + "normalize_mu.npy"
    sigma_dir = data_dir + "normalize_sigma.npy"
    np.save(mu_dir, mu)
    np.save(sigma_dir, sigma)
    '''
    return x_scaled, mu, sigma


def normalize_df(df, data_dir):
    column_names = df.columns
    x = df.values
    x_scaled, mu, sigma = normalize(x)
    mu = mu.reshape(1, -1)
    sigma = sigma.reshape(1, -1)

    normalize_mu = pd.DataFrame(data=mu, columns=column_names)
    normalize_sigma = pd.DataFrame(data=sigma, columns=column_names)
    normalize_param = pd.concat([normalize_mu, normalize_sigma])
    normalize_param.to_csv(data_dir + "normalize_param.csv")
    scaled_df = pd.DataFrame(data=x_scaled, columns=column_names)
    return scaled_df


def digitize_columns(data_frame):
    print("[INFO] digitize columns")

    data_frame = data_frame.replace({"work_year": work_year_map, "class": class_map})
    return data_frame


def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'


def prepare_data(file_path):
    print("[INFO] prepare loan data.")

    df_loan = pd.read_csv(file_path, low_memory=False)
    # print(f"[INFO] loaded loan data with shape:{df_loan.shape} to :{file_path}")
    df_loan = digitize_columns(df_loan)

    df_loan['issue_date'] = pd.to_datetime(df_loan['issue_date'])

    df_loan['issue_date_month'] = df_loan['issue_date'].dt.month

    df_loan['issue_date_dayofweek'] = df_loan['issue_date'].dt.dayofweek

    cols = ['employer_type', 'industry']
    for col in cols:
        lbl = LabelEncoder().fit(df_loan[col])
        df_loan[col] = lbl.transform(df_loan[col])

    df_loan['earlies_credit_mon'] = pd.to_datetime(df_loan['earlies_credit_mon'].map(findDig))

    df_loan['earliesCreditMon'] = df_loan['earlies_credit_mon'].dt.month
    df_loan['earliesCreditYear'] = df_loan['earlies_credit_mon'].dt.year

    df_loan.fillna(method='bfill', inplace=True)

    col_to_drop = ['issue_date', 'earlies_credit_mon']
    df_loan = df_loan.drop(col_to_drop, axis=1)

    return df_loan


def process_data(loan_df, data_dir):
    loan_feat_df = loan_df[all_feature_list]

    norm_loan_feat_df = normalize_df(loan_feat_df, data_dir)

    loan_target_df = loan_df[['isDefault']]
    loan_target = loan_target_df.values
    reindex_loan_target_df = pd.DataFrame(loan_target, columns=loan_target_df.columns)
    processed_loan_df = pd.concat([norm_loan_feat_df, reindex_loan_target_df], axis=1)
    # processed_loan_df = pd.concat([loan_feat_df, loan_target_df], axis=1)
    return processed_loan_df


def load_processed_data(data_dir):
    file_path = data_dir + "processed_loan.csv"
    if os.path.exists(file_path):
        print(f"[INFO] load processed loan data from {file_path}")
        processed_loan_df = pd.read_csv(file_path, low_memory=False)
    else:
        # print(f"[INFO] start processing loan data.")
        file_path = data_dir + "train_public.csv"
        processed_loan_df = process_data(prepare_data(file_path), data_dir)
        # processed_loan_df = process_unnormalized_data(prepare_data(file_path))
        file_path = data_dir + "processed_loan.csv"
        processed_loan_df.to_csv(file_path, index=False)
        print(f"[INFO] save processed loan data to: {file_path}")
    return processed_loan_df


def loan_load_two_party_data(data_dir):
    print("[INFO] load two party data")
    processed_loan_df = load_processed_data(data_dir)
    processed_loan_df = shuffle(processed_loan_df)

    processed_loan_feature = processed_loan_df.drop('isDefault', axis=1)
    feature_columns = processed_loan_feature.columns.tolist()
    party_b_feat_list = random.sample(feature_columns, 19)

    party_a_feat_list = [n for n in feature_columns if n not in party_b_feat_list]

    # party_a_feat_list = qualification_feat + loan_feat
    # party_b_feat_list = debt_feat + repayment_feat
    '''
    X, Y = processed_loan_df[all_feature_list], processed_loan_df['isDefault']

    smo = SMOTE(sampling_strategy=0.25, random_state=42)
    X_smo, Y_smo = smo.fit_resample(X, Y)

    Xa, Xb, y = X_smo[party_a_feat_list].values, X_smo[party_b_feat_list].values, Y_smo.values
    '''
    Xa, Xb, y = processed_loan_df[party_a_feat_list].values, processed_loan_df[party_b_feat_list].values, \
        processed_loan_df['isDefault'].values

    y = np.expand_dims(y, axis=1)

    n_train = int(0.9 * Xa.shape[0])

    print("# of train samples:", n_train)

    Xa_train, Xb_train = Xa[:n_train], Xb[:n_train]

    Xa_test, Xb_test = Xa[n_train:], Xb[n_train:]

    # Xa_attn, Xb_attn = Xa[:n_test], Xb[:n_test]
    # y_attn = y[:n_test]
    # for re-train model and query for cf

    y_train, y_test = y[:n_train], y[n_train:]

    # take one example for cf learning
    # Xa_query, Xb_query, y_query = Xa_query[100].reshape(1,-1), Xb_query[100].reshape(1,-1), y_query[100].reshape(1,-1)

    # take a trained instance for cf learning
    Xa_query, Xb_query, y_query = Xa_train[20].reshape(1, -1), Xb_train[20].reshape(1, -1), y_train[20].reshape(1, -1)

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xa_query.shape:", Xa_query.shape)
    print("Xb_query.shape:", Xb_query.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_query.shape:", y_query.shape)
    print("y_test.shape:", y_test.shape, type(y_test))

    train_prototype = processed_loan_df[:n_train]

    reject_prototype = train_prototype[train_prototype['isDefault'] == 1]
    grant_prototype = train_prototype[train_prototype['isDefault'] == 0]

    Xa_rej, Xb_rej, y_reg = reject_prototype[party_a_feat_list].values, reject_prototype[party_b_feat_list].values, \
        reject_prototype['isDefault'].values

    Xa_gre, Xb_gre, y_gre = grant_prototype[party_a_feat_list].values, grant_prototype[party_b_feat_list].values, \
        grant_prototype['isDefault'].values

    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]


def poisoned_loan_load_two_party_data(data_dir):
    print("[INFO] load two party data")
    processed_loan_df = load_processed_data(data_dir)
    processed_loan_df = shuffle(processed_loan_df)

    party_a_feat_list = qualification_feat + loan_feat
    party_b_feat_list = debt_feat + repayment_feat

    X, Y = processed_loan_df[all_feature_list], processed_loan_df['isDefault']
    smo = SMOTE(sampling_strategy=0.25, random_state=42)
    X_smo, Y_smo = smo.fit_resample(X, Y)

    Xa, Xb, y = X_smo[party_a_feat_list].values, X_smo[party_b_feat_list].values, Y_smo.values

    # Xa, Xb, y = processed_loan_df[party_a_feat_list].values, processed_loan_df[party_b_feat_list].values, \
    #            processed_loan_df['isDefault'].values

    y = np.expand_dims(y, axis=1)

    total_num = Xa.shape[0]

    n_train = int(0.8 * total_num)

    print("# of train samples:", n_train)

    Xa_train, Xb_train = Xa[:n_train], Xb[:n_train]

    Xa_test, Xb_test = Xa[n_train:], Xb[n_train:]

    y_train, y_test = y[:n_train], y[n_train:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)

    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))

    train_prototype = processed_loan_df[:n_train]

    Dtarget = train_prototype[train_prototype['isDefault'] == 0][:10]

    Xa_DT, Xb_DT, y_DT = Dtarget[party_a_feat_list].values, Dtarget[party_b_feat_list].values, \
        Dtarget['isDefault'].values

    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test], [Xa_DT, Xb_DT, y_DT]


def loan_load_three_party_data(data_dir):
    print("[INFO] load three party data")
    processed_loan_df = load_processed_data(data_dir)
    party_a_feat_list = qualification_feat + loan_feat
    party_b_feat_list = debt_feat + repayment_feat
    party_c_feat_list = multi_acc_feat + mal_behavior_feat
    Xa, Xb, Xc, y = processed_loan_df[party_a_feat_list].values, processed_loan_df[party_b_feat_list].values, \
        processed_loan_df[party_c_feat_list].values, processed_loan_df['target'].values

    y = np.expand_dims(y, axis=1)
    n_train = int(0.8 * Xa.shape[0])
    Xa_train, Xb_train, Xc_train = Xa[:n_train], Xb[:n_train], Xc[:n_train]
    Xa_test, Xb_test, Xc_test = Xa[n_train:], Xb[n_train:], Xc[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xc_train.shape:", Xc_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("Xc_test.shape:", Xc_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape)
    return [Xa_train, Xb_train, Xc_train, y_train], [Xa_test, Xb_test, Xc_test, y_test]


if __name__ == '__main__':
    data_dir = "../../../data/zhongyuan/"
    loan_load_two_party_data(data_dir)
