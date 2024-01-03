# Python version 3.11.3

import pandas as pd
import numpy as np
import category_encoders as ce

from imblearn.under_sampling import RandomUnderSampler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import AUC

import pickle

np.random.seed(42)

pd.set_option('display.max_columns', None)

#----------------------------------------------------------------------------------

def read_file(path):
    """This function is designed to read a CSV file located at a specified path."""
    df = pd.read_csv(path)
    return df

#----------------------------------------------------------------------------------

def balance_class(df):
    """This function balances the distribution of the target class."""
    x = df.drop(columns="TARGET")
    y = df["TARGET"]

    rus = RandomUnderSampler(random_state=0)
    loan, target = rus.fit_resample(X=x, y=y)
    df_balanced = loan.join(target)
    df_balanced = df_balanced.sample(frac=1)
    df_balanced = df_balanced.reset_index(drop=True)
    return df_balanced

#----------------------------------------------------------------------------------

def remove_outliers(df):
    """This function eliminates outliers from the dataset."""
    outliers = df[(df['AMT_INCOME_TOTAL']>15000000)].index
    df = df.drop(index=outliers).reset_index(drop=True)
    return df

#----------------------------------------------------------------------------------

def days_features(df):
    """This function converts features representing days into their equivalent in years."""
    df['AGE'] = df['DAYS_BIRTH']/-365
    df['TENURE'] = df['DAYS_EMPLOYED']/-365
    df['DAYS_REGISTRATION'] = df['DAYS_REGISTRATION']/-365
    df['DAYS_ID_PUBLISH'] = df['DAYS_ID_PUBLISH']/-365
    df['DAYS_LAST_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE']/-365
    df['TENURE'] = np.where((df['TENURE']<0), 0, df['TENURE'])
    return df

#----------------------------------------------------------------------------------

def ext_source_knn(df, feature_list, external_source):
    """This function utilizes the KNeighbors algorithm to predict values for the external sources features."""
    ext_features = df[feature_list + external_source]
    ext_features_nn = ext_features.dropna()

    x_ext = ext_features_nn[feature_list]
    y_ext = ext_features_nn[external_source]

    ext_knn = KNeighborsRegressor(n_neighbors=5)
    ext_knn.fit(x_ext, y_ext)

    source = ''.join(external_source)
    ext_null = ext_features[ext_features[source].isna()].iloc[:, :len(feature_list)]
    ext_pred = ext_knn.predict(ext_null)
    return ext_pred

#----------------------------------------------------------------------------------

def fill_missing_external(df, column_name, pred_list):
    """This function completes the missing values in the external sources features that were predicted using the KNeighbors algorithm."""
    missing_indices = df[df[column_name].isnull()].index

    for i, index in enumerate(missing_indices):
        if i < len(pred_list):
            df.at[index, column_name] = pred_list[i]
        else:
            break
    return df

#----------------------------------------------------------------------------------

def data_imputation(df):
    """This function serves to impute missing values within the dataset."""
    df['OWN_CAR_AGE'] = np.where(((df['FLAG_OWN_CAR']=='N')&(df['OWN_CAR_AGE'].isna())), 0, df['OWN_CAR_AGE'])
    car_age_med = df[df['FLAG_OWN_CAR']=='Y']['OWN_CAR_AGE'].median()
    df['OWN_CAR_AGE'] = np.where(((df['FLAG_OWN_CAR']=='Y')&(df['OWN_CAR_AGE'].isna())), car_age_med, df['OWN_CAR_AGE'])
    df['log_INCOME_TOTAL'] = np.log(df['AMT_INCOME_TOTAL'])

    df['OCCUPATION_TYPE'] = np.where(((df['OCCUPATION_TYPE'].isna())&(df['NAME_INCOME_TYPE']=='Pensioner')&(df['ORGANIZATION_TYPE']=='XNA')), "XNA", df['OCCUPATION_TYPE'])
    df['OCCUPATION_TYPE'] = np.where(((df['OCCUPATION_TYPE'].isna())&(df['NAME_INCOME_TYPE']=='Pensioner')&(df['ORGANIZATION_TYPE']=='Trade: type 7')), "Sales staff", df['OCCUPATION_TYPE'])
    df['OCCUPATION_TYPE'] = np.where(((df['OCCUPATION_TYPE'].isna())&(df['NAME_INCOME_TYPE']=='Unemployed')&(df['ORGANIZATION_TYPE']=='XNA')), "XNA", df['OCCUPATION_TYPE'])
    df['OCCUPATION_TYPE'] = np.where(((df['OCCUPATION_TYPE'].isna())&(df['NAME_INCOME_TYPE']=='Working')), "Laborers", df['OCCUPATION_TYPE'])
    df['OCCUPATION_TYPE'] = np.where(((df['OCCUPATION_TYPE'].isna())&(df['NAME_INCOME_TYPE']=='Commercial associate')), "Laborers", df['OCCUPATION_TYPE'])
    df['OCCUPATION_TYPE'] = np.where(((df['OCCUPATION_TYPE'].isna())&(df['NAME_INCOME_TYPE']=='State servant')), "Core staff", df['OCCUPATION_TYPE'])
    df['OCCUPATION_TYPE'] = np.where(((df['OCCUPATION_TYPE'].isna())&(df['NAME_INCOME_TYPE']=='State servant')), "Core staff", df['OCCUPATION_TYPE'])
    df['NAME_INCOME_TYPE'] = np.where((df['NAME_INCOME_TYPE']=='Student'), "Working", df['NAME_INCOME_TYPE'])
    df['OCCUPATION_TYPE'] = np.where(((df['OCCUPATION_TYPE'].isna())&(df['NAME_INCOME_TYPE']=='Working')), "Other", df['OCCUPATION_TYPE'])

    df['AMT_REQ_CREDIT_BUREAU_HOUR'] = np.where(((df['TENURE']==0)&(df['ORGANIZATION_TYPE']=='XNA')), 0, df['AMT_REQ_CREDIT_BUREAU_HOUR'])
    df['AMT_REQ_CREDIT_BUREAU_DAY'] = np.where(((df['TENURE']==0)&(df['ORGANIZATION_TYPE']=='XNA')), 0, df['AMT_REQ_CREDIT_BUREAU_DAY'])
    df['AMT_REQ_CREDIT_BUREAU_WEEK'] = np.where(((df['TENURE']==0)&(df['ORGANIZATION_TYPE']=='XNA')), 0, df['AMT_REQ_CREDIT_BUREAU_WEEK'])
    df['AMT_REQ_CREDIT_BUREAU_MON'] = np.where(((df['TENURE']==0)&(df['ORGANIZATION_TYPE']=='XNA')), 0, df['AMT_REQ_CREDIT_BUREAU_MON'])
    df['AMT_REQ_CREDIT_BUREAU_QRT'] = np.where(((df['TENURE']==0)&(df['ORGANIZATION_TYPE']=='XNA')), 0, df['AMT_REQ_CREDIT_BUREAU_QRT'])
    df['AMT_REQ_CREDIT_BUREAU_YEAR'] = np.where(((df['TENURE']==0)&(df['ORGANIZATION_TYPE']=='XNA')), 0, df['AMT_REQ_CREDIT_BUREAU_YEAR'])

    df['NAME_HOUSING_TYPE'] = np.where(((df['HOUSETYPE_MODE'].isna())&(df['NAME_HOUSING_TYPE']=='House / apartment')), 'House', df['NAME_HOUSING_TYPE'])
    df['HOUSETYPE_MODE'] = np.where((df['HOUSETYPE_MODE'].isna()), 'house', df['HOUSETYPE_MODE'])

    df['FONDKAPREMONT_MODE'] = np.where((df['HOUSETYPE_MODE']=='house'), 'XNA', df['FONDKAPREMONT_MODE'])
    df['WALLSMATERIAL_MODE'] = np.where((df['HOUSETYPE_MODE']=='house'), 'XNA', df['WALLSMATERIAL_MODE'])
    df['EMERGENCYSTATE_MODE'] = np.where((df['HOUSETYPE_MODE']=='house'), 'XNA', df['EMERGENCYSTATE_MODE'])

    df['FONDKAPREMONT_MODE'] = np.where(((df['FONDKAPREMONT_MODE'].isna())&(df['HOUSETYPE_MODE']!='house')), 'not specified', df['FONDKAPREMONT_MODE'])
    df['WALLSMATERIAL_MODE'] = np.where(((df['WALLSMATERIAL_MODE'].isna())&(df['HOUSETYPE_MODE']=='specific housing')), 'Stone, brick', df['WALLSMATERIAL_MODE'])
    df['WALLSMATERIAL_MODE'] = np.where(((df['WALLSMATERIAL_MODE'].isna())&(df['HOUSETYPE_MODE']=='terraced house')), 'Stone, brick', df['WALLSMATERIAL_MODE'])

    df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].fillna(1.0)
    df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].fillna('Unknown')
    df['AMT_GOODS_PRICE'] = np.where((df['AMT_GOODS_PRICE'].isna()), df['AMT_CREDIT'], df['AMT_GOODS_PRICE'])
    df['WALLSMATERIAL_MODE'] = df['WALLSMATERIAL_MODE'].fillna('Unknown')
    df['AMT_ANNUITY'] = df['AMT_ANNUITY'].fillna(27652.5) #median of the amount annuity when the credit amount is 539,100
    df = df.fillna(-1)

    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map(lambda x: 1 if x=='Y' else 0)
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map(lambda x: 1 if x=='Y' else 0)

    df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].map(lambda x: 'Apartment' if x=='House / apartment' else x)

    return df

#----------------------------------------------------------------------------------

def drop_features(df):
    """"This function facilitates dropping unused features from the dataset for model training."""
    df = df.drop(columns=['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'DAYS_BIRTH', 'DAYS_EMPLOYED','FLAG_DOCUMENT_12'])
    return df

#----------------------------------------------------------------------------------

def merge_datasets(df, data1, data2, data3, data4):
    """This function merges multiple datasets, including 'bureau.csv', 'credit.csv', 'installment.csv', and 'POS.csv', into a single combined dataset."""
    final_train = df.merge(data1, on='SK_ID_CURR', how='left')
    final_train = final_train.merge(data2, on='SK_ID_CURR', how='left')
    final_train = final_train.merge(data3, on='SK_ID_CURR', how='left')
    final_train = final_train.merge(data4, on='SK_ID_CURR', how='left')
    final_train['LAST_STATUS'] = final_train['LAST_STATUS'].fillna('Unknown')
    final_train = final_train.fillna(0)
    return final_train

#----------------------------------------------------------------------------------

def feature_engineering(df):
    """This function focuses on creating new features or transforming existing ones to enhance the dataset's predictive power or improve model performance."""
    income_avg = df['log_INCOME_TOTAL'].mean()
    income_std = df['log_INCOME_TOTAL'].std()

    lower_limit = income_avg-(1*income_std)
    upper_limit = income_avg+(1*income_std)
    upper_limit2 = income_avg+(2*income_std)

    df['CRED_INCOME_RATIO'] = round(df['AMT_CREDIT']/df['AMT_INCOME_TOTAL']*100, 4)
    df['GOODS_ANNUITY_RATIO'] = round(df['AMT_GOODS_PRICE']/df['AMT_ANNUITY']*100, 4)
    df['ANNUITY_INCOME_RATIO'] = round(df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL']*100, 4)
    df['CREDIT_USAGE'] = round(df['AMT_GOODS_PRICE']/df['AMT_CREDIT']*100, 4)

    df['INCOME_LEVEL'] = df['log_INCOME_TOTAL'].map(lambda x: 'High' if x >= upper_limit2 else 'Above Avg' if x>= upper_limit else 'Below Avg' if x<=lower_limit else 'Low')

    df['TENURE_AGE_RATIO'] = round((df['TENURE']/df['AGE']), 4)
    df['OWN_CAR_RATIO'] = round((df['OWN_CAR_AGE']/df['AGE']), 4)

    df['ADDRESS_REGION_FLAG'] = df['REG_REGION_NOT_LIVE_REGION'] + df['REG_REGION_NOT_WORK_REGION'] + df['LIVE_REGION_NOT_WORK_REGION']
    df['ADDRESS_REGION_FLAG'] = df['ADDRESS_REGION_FLAG'].map(lambda x: 1 if x > 0 else x)
    df['ADDRESS_CITY_FLAG'] = df['REG_CITY_NOT_LIVE_CITY'] + df['REG_CITY_NOT_WORK_CITY'] + df['LIVE_CITY_NOT_WORK_CITY']
    df['ADDRESS_CITY_FLAG'] = df['ADDRESS_CITY_FLAG'].map(lambda x: 1 if x > 0 else x)

    df['LEFT_INCOME'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']

    df['MONTHLY_INCOME'] = df['AMT_INCOME_TOTAL']/12
    df['MONTHLY_INSTALLMENT'] = df['AMT_ANNUITY']/12

    df['DTI'] = df['MONTHLY_INSTALLMENT']/df['MONTHLY_INCOME']
    df['HOUR_APPR_PROCESS_START'] = df['HOUR_APPR_PROCESS_START'].astype('object')

    df = df.drop(columns=['AMT_INCOME_TOTAL','REG_REGION_NOT_LIVE_REGION','REG_CITY_NOT_LIVE_CITY','REG_REGION_NOT_WORK_REGION','REG_CITY_NOT_WORK_CITY','LIVE_REGION_NOT_WORK_REGION','LIVE_CITY_NOT_WORK_CITY'])

    return df

#----------------------------------------------------------------------------------

def scale_data(df):
    """This function standardizes the numeric data by employing the MinMaxScaler to bring all values within a defined range, typically between 0 and 1."""
    numeric_features = df[df.drop(columns=['SK_ID_CURR', 'TARGET']).select_dtypes(exclude='object').columns]
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(numeric_features), columns=numeric_features.columns)
    return df_scaled

#----------------------------------------------------------------------------------

def encode_data(df):
    """
This function utilizes the OneHotEncoder to transform categorical data into a format suitable for training algorithms by converting categorical variables into binary vectors."""
    encode_list = df.select_dtypes('object').columns
    cat_features = df[encode_list]
    onehot = ce.OneHotEncoder(use_cat_names=True)
    train_encoded = onehot.fit_transform(cat_features)
    return train_encoded

#----------------------------------------------------------------------------------

def generate_final_data(scaled_df, encoded_df):
    """This function consolidates the scaled and encoded data, preparing it for final training before implementing machine learning algorithms."""
    train_final = scaled_df.join(encoded_df)
    return train_final

#----------------------------------------------------------------------------------

def split_data(train, target):
    """This function divides the dataset into training and testing subsets for model evaluation and validation."""
    x = train
    y = target['TARGET']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

#----------------------------------------------------------------------------------

def train_pred_model(x_train, y_train, x_test, model):
    """This function employs the trained model to predict outcomes on the validation dataset, providing both binary and probability results."""
    model = model.fit(X=x_train, y=y_train)
    model_pred = model.predict(x_test)
    model_pred_proba = model.predict_proba(x_test)
    return model_pred, model_pred_proba

#----------------------------------------------------------------------------------

def print_scores(true_label, prediction):
    """This function calculates precision, recall, and area under the curve scores for model evaluation, offering insights into its performance."""
    score_data = {
    'Precision': [precision_score(true_label, prediction)],
    'Recall': [recall_score(true_label, prediction)],
    'ROC_AUC':[roc_auc_score(true_label, prediction)]
    }

    score_df = pd.DataFrame(score_data)
    return score_df

#----------------------------------------------------------------------------------

def train_ann(x_train, y_train):
    """"This function handles the training process for artificial neural networks."""
    ann_model = Sequential()
    ann_model.add(Dense(64, input_dim=315, activation='relu'))
    ann_model.add(Dense(32, activation='relu'))
    ann_model.add(Dense(32, activation='relu'))
    ann_model.add(Dense(32, activation='relu'))
    ann_model.add(Dense(1, activation='sigmoid'))

    opt = Adam(learning_rate=0.0001)
    callback = EarlyStopping(monitor='val_loss', patience=10)

    ann_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[AUC()])

    history = ann_model.fit(x_train, y_train, batch_size=1000, epochs=100, validation_split=0.2, callbacks=[callback])
    return ann_model

#----------------------------------------------------------------------------------

def predict_ann(x_test, model):
    """"This function leverages the trained artificial neural network model to generate predictions."""
    ann_pred = model.predict(x_test)
    return ann_pred

#----------------------------------------------------------------------------------