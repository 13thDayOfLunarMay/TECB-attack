qualification_feat = [
    'total_loan',
    'class',
    'employer_type',
    'industry',
    'work_year',
    'house_exist',
    'censor_status',
    'post_code',
    'region']



loan_feat = [
    'year_of_loan',
    'interest',
    'monthly_payment',
    'use',
    'app_type',
    'title']

debt_feat = [
    'debt_loan_ratio',
    'del_in_18month',
    'scoring_low',
    'scoring_high',
    'known_outstanding_loan',
    'known_dero',
    'pub_dero_bankrup',
    'initial_list_status',
    'issue_date_month',
    'issue_date_dayofweek',
    'earliesCreditMon',
    'earliesCreditYear']

repayment_feat = [
    'recircle_b',
    'recircle_u',
    'f0',
    'f1',
    'f2',
    'f3',
    'f4',
    'early_return',
    'early_return_amount',
    'early_return_amount_3mon']


mask_qualification_feat = ['total_loan',
    'class',
    'employer_type',
    'industry',
    'work_year',
    'house_exist',
    'region']

mask_loan_feat = [
    'year_of_loan',
    'interest',
    'monthly_payment',
    'use',
    'title']



mask_debt_feat = [
    'debt_loan_ratio',
    'del_in_18month',
    'scoring_low',
    'scoring_high',
    'known_outstanding_loan',
    'pub_dero_bankrup',
    'initial_list_status',
    'issue_date_month',
    'issue_date_dayofweek',
    'earliesCreditMon',
    'earliesCreditYear']

mask_repayment_feat = [
    'recircle_u',
    'f0',
    'f2',
    'f3',
    'f4',
    'early_return',
    'early_return_amount',
    'early_return_amount_3mon']


all_feature_list = qualification_feat + loan_feat + debt_feat + repayment_feat

