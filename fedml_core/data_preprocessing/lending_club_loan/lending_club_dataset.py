import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

from .lending_club_feature_group import all_feature_list, qualification_feat, \
    loan_feat, debt_feat, repayment_feat, multi_acc_feat, mal_behavior_feat

target_map = {'Good Loan': 0, 'Bad Loan': 1}

grade_map = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
sub_grade = ['C1' 'D2' 'D1' 'C4' 'C3' 'C2' 'D5' 'B3' 'A4' 'B5' 'C5' 'D4' 'E1' 'E4'
             'B4' 'D3' 'A1' 'E5' 'B2' 'B1' 'A5' 'F5' 'A3' 'E3' 'A2' 'E2' 'F4' 'G1'
             'G2' 'F1' 'F2' 'F3' 'G4' 'G3' 'G5']
emp_length_map = {np.nan: 0, '< 1 year': 1, '1 year': 2, '2 years': 2, '3 years': 2, '4 years': 3, '5 years': 3,
                  '6 years': 3, '7 years': 4, '8 years': 4, '9 years': 4, '10+ years': 5}

home_ownership_map = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'ANY': 3, 'NONE': 3, 'OTHER': 3}

verification_status_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}

# verification_status_joint = [nan 'Verified' 'Not Verified' 'Source Verified']

term_map = {' 36 months': 0, ' 60 months': 1}
initial_list_status_map = {'w': 0, 'f': 1}
purpose_map = {'debt_consolidation': 0, 'credit_card': 0, 'small_business': 1, 'educational': 2,
               'car': 3, 'other': 3, 'vacation': 3, 'house': 3, 'home_improvement': 3, 'major_purchase': 3,
               'medical': 3, 'renewable_energy': 3, 'moving': 3, 'wedding': 3}
application_type_map = {'Individual': 0, 'Joint App': 1}
disbursement_method_map = {'Cash': 0, 'DirectPay': 1}

'''
def normalize(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled
'''

def normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_scaled = (x-mu)/sigma
    '''
    mu_dir = data_dir + "normalize_mu.npy"
    sigma_dir = data_dir + "normalize_sigma.npy"
    np.save(mu_dir, mu)
    np.save(sigma_dir, sigma)
    '''
    return x_scaled, mu, sigma

def cf_denormalize(loan, query_scaled, cf_scaled):
    scaler = StandardScaler()

    loan_norm = scaler.fit_transform(loan)

    query = scaler.inverse_transform(query_scaled)
    cf = scaler.inverse_transform(cf_scaled)
    return query, cf

def normalize_df(df, data_dir):
    column_names = df.columns
    x = df.values
    x_scaled, mu, sigma = normalize(x)
    mu = mu.reshape(1,-1)
    sigma = sigma.reshape(1,-1)

    normalize_mu = pd.DataFrame(data=mu, columns=column_names)
    normalize_sigma = pd.DataFrame(data=sigma, columns=column_names)
    normalize_param = pd.concat([normalize_mu, normalize_sigma])
    normalize_param.to_csv(data_dir+"normalize_param.csv")
    scaled_df = pd.DataFrame(data=x_scaled, columns=column_names)
    return scaled_df


def loan_condition(status):
    bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period",
                "Late (16-30 days)", "Late (31-120 days)"]
    if status in bad_loan:
        return 'Bad Loan'
    else:
        return 'Good Loan'


def compute_annual_income(row):
    if row['verification_status'] == row['verification_status_joint']:
        return row['annual_inc_joint']
    return row['annual_inc']


def determine_good_bad_loan(df_loan):
    print("[INFO] determine good or bad loan")

    df_loan['target'] = np.nan
    df_loan['target'] = df_loan['loan_status'].apply(loan_condition)
    return df_loan


def determine_annual_income(df_loan):
    print("[INFO] determine annual income")

    df_loan['annual_inc_comp'] = np.nan
    df_loan['annual_inc_comp'] = df_loan.apply(compute_annual_income, axis=1)
    return df_loan


def determine_issue_year(df_loan):
    print("[INFO] determine issue year")

    # transform the issue dates by year
    dt_series = pd.to_datetime(df_loan['issue_d'])
    df_loan['issue_year'] = dt_series.dt.year
    return df_loan


def digitize_columns(data_frame):
    print("[INFO] digitize columns")

    data_frame = data_frame.replace({"target": target_map, "grade": grade_map, "emp_length": emp_length_map,
                                     "home_ownership": home_ownership_map,
                                     "verification_status": verification_status_map,
                                     "term": term_map, "initial_list_status": initial_list_status_map,
                                     "purpose": purpose_map, "application_type": application_type_map,
                                     "disbursement_method": disbursement_method_map})
    return data_frame


def prepare_data(file_path):
    print("[INFO] prepare loan data.")

    df_loan = pd.read_csv(file_path, low_memory=False)
    # print(f"[INFO] loaded loan data with shape:{df_loan.shape} to :{file_path}")

    df_loan = determine_good_bad_loan(df_loan)
    df_loan = determine_annual_income(df_loan)
    df_loan = determine_issue_year(df_loan)
    df_loan = digitize_columns(df_loan)

    df_loan = df_loan[df_loan['issue_year'] == 2018]
    return df_loan


def process_data(loan_df, data_dir):
    loan_feat_df = loan_df[all_feature_list]

    loan_feat_df = loan_feat_df.fillna(-99)
    assert loan_feat_df.isnull().sum().sum() == 0
    norm_loan_feat_df = normalize_df(loan_feat_df, data_dir)


    loan_target_df = loan_df[['target']]
    loan_target = loan_target_df.values
    reindex_loan_target_df = pd.DataFrame(loan_target, columns= loan_target_df.columns)
    processed_loan_df = pd.concat([norm_loan_feat_df, reindex_loan_target_df], axis=1)
    #processed_loan_df = pd.concat([loan_feat_df, loan_target_df], axis=1)
    return processed_loan_df


def process_unnormalized_data(loan_df):
    loan_feat_df = loan_df[all_feature_list]

    '''
    loan_feat_df = loan_feat_df.fillna(-99)
    assert loan_feat_df.isnull().sum().sum() == 0
    norm_loan_feat_df = normalize_df(loan_feat_df)
    '''

    loan_target_df = loan_df[['target']]
    #loan_target = loan_target_df.values
    #reindex_loan_target_df = pd.DataFrame(loan_target, columns=loan_target_df.columns)
    #processed_loan_df = pd.concat([norm_loan_feat_df, reindex_loan_target_df], axis=1)
    processed_loan_df = pd.concat([loan_feat_df, loan_target_df], axis=1)
    return processed_loan_df

def load_processed_data(data_dir):
    file_path = data_dir + "processed_loan.csv"
    if os.path.exists(file_path):
        print(f"[INFO] load processed loan data from {file_path}")
        processed_loan_df = pd.read_csv(file_path, low_memory=False)
    else:
        # print(f"[INFO] start processing loan data.")
        file_path = data_dir + "accepted_2007_to_2018Q4.csv"
        processed_loan_df = process_data(prepare_data(file_path), data_dir)
        #processed_loan_df = process_unnormalized_data(prepare_data(file_path))
        file_path = data_dir + "processed_loan.csv"
        processed_loan_df.to_csv(file_path, index=False)
        print(f"[INFO] save processed loan data to: {file_path}")
    return processed_loan_df




def loan_load_two_party_data(data_dir):
    print("[INFO] load two party data")
    #random.seed(2)
    processed_loan_df = load_processed_data(data_dir)
    processed_loan_feature = processed_loan_df.drop('target', axis=1)
    feature_columns=processed_loan_feature.columns.tolist()
    party_b_feat_list = random.sample(feature_columns, 42)

    party_a_feat_list = [n for n in feature_columns if n not in party_b_feat_list]

    #party_a_feat_list = qualification_feat + loan_feat
    #party_b_feat_list = debt_feat + repayment_feat + multi_acc_feat + mal_behavior_feat

    Xa, Xb, y = processed_loan_df[party_a_feat_list].values, processed_loan_df[party_b_feat_list].values, \
                processed_loan_df['target'].values
    y = np.expand_dims(y, axis=1)
    n_train = int(0.8 * Xa.shape[0])
    print("# of train samples:", n_train)

    Xa_train, Xb_train = Xa[:n_train], Xb[:n_train]

    #n_test = int(0.9 * Xa.shape[0])

    Xa_test, Xb_test = Xa[n_train:], Xb[n_train:]

    #Xa_attn, Xb_attn = Xa[:n_test], Xb[:n_test]
    #y_attn = y[:n_test]
    # for re-train model and query for cf
    #Xa_query, Xb_query = Xa[n_test:], Xb[n_test:]

    y_train,  y_test = y[:n_train], y[n_train:]

    #take one example for cf learning
    #Xa_query, Xb_query, y_query = Xa_query[100].reshape(1,-1), Xb_query[100].reshape(1,-1), y_query[100].reshape(1,-1)

    #take a trained instance for cf learning
    #Xa_query, Xb_query, y_query = Xa_train[20].reshape(1, -1), Xb_train[20].reshape(1, -1), y_train[20].reshape(1, -1)

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))

    '''
    print("Xa_query.shape:", Xa_query.shape)
    print("Xb_query.shape:", Xb_query.shape)
    print("y_query.shape:", y_query.shape)

    train_prototype = processed_loan_df[:n_train]

    reject_prototype = train_prototype[train_prototype['target']==1]
    grant_prototype = train_prototype[train_prototype['target'] == 0]

    Xa_rej, Xb_rej, y_reg = reject_prototype[party_a_feat_list].values, reject_prototype[party_b_feat_list].values, \
                reject_prototype['target'].values

    Xa_gre, Xb_gre, y_gre = grant_prototype[party_a_feat_list].values, grant_prototype[party_b_feat_list].values, \
                            grant_prototype['target'].values
    '''


    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]




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
    data_dir = "../../../data/lending_club_loan/"
    loan_load_two_party_data(data_dir)
