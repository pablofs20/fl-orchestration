import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split

NULL_THRESHOLD = 0.3


def remove_high_null_vars(df):
    n_columns = len(df.columns)

    null_pctg = df.isnull().sum() / len(df)
    columns_to_drop = null_pctg[null_pctg > NULL_THRESHOLD].index
    df = df.drop(columns_to_drop, axis=1).reset_index(drop=True)

    print("{cd} columns (from {total}) are dropped since their null values percentage excess {th}% threshold".format(
        cd=len(columns_to_drop), total=n_columns, th=NULL_THRESHOLD*100))

    return df


def remove_high_null_samples(df):
    n_rows = len(df.index)

    null_pctg = df.isnull().sum(axis=1) / df.shape[1]
    rows_to_drop = null_pctg[null_pctg > NULL_THRESHOLD].index
    df = df.drop(rows_to_drop, axis=0).reset_index(drop=True)

    print("{rd} rows (from {total}) are dropped since their null values percentage excess {th}% threshold".format(
        rd=len(rows_to_drop), total=n_rows, th=NULL_THRESHOLD*100))
    
    return df


def imput_nan_values(df):
    # Separamos el dataframe en 2 versiones: uno con las variables categóricas y otro con las numéricas
    num_cols = df._get_numeric_data().columns

    cat_cols = df.select_dtypes(include=['category']).columns

    df_num = df[num_cols]
    df_cat = df[cat_cols]

    # Imputamos NaN usando K-NN con 3 vecinos sobre las variables numéricas
    imputer = KNNImputer(n_neighbors=3, weights="uniform")

    print("imputing")
    print(num_cols)
    print(df_num)
    df_num_imputed = pd.DataFrame(
        imputer.fit_transform(df_num), columns=num_cols)

    print("imputed")
    # Imputamos NaN usando SimpleImputer reemplazando por el valor más frecuente sobre las variables categóricas
    imputer = SimpleImputer(strategy='most_frequent')
    df_cat_imputed = pd.DataFrame(
        imputer.fit_transform(df_cat), columns=cat_cols)


    print("NaN values on numerical and categorical values have been imputed")
    return df_num_imputed, df_cat_imputed


def remove_correlated_vars(df_num_imputed):
    correlation_matrix = df_num_imputed.corr().abs()

    correlated_vars = correlation_matrix[correlation_matrix > 0.9].stack(
    ).reset_index()
    correlated_vars = correlated_vars[correlated_vars['level_0']
                                      != correlated_vars['level_1']]
    correlated_vars.columns = ['Variable 1', 'Variable 2', 'Correlation']

    var_list = np.unique(list(correlated_vars["Variable 1"]))

    columns_to_drop = set()

    for var in var_list:
        if var not in columns_to_drop:
            a = correlated_vars[correlated_vars['Variable 1']
                                == var]['Variable 2'].tolist()
            ok_list = correlated_vars.loc[~correlated_vars['Variable 1'].isin(
                a)]['Variable 1'].unique()

            diff = [x for x in var_list if x not in ok_list]
            columns_to_drop.update(diff)

    columns_to_drop = list(columns_to_drop)

    print("The following variables are going to be dropped since they are correlated with others:")
    print(columns_to_drop)

    df_num_imputed = df_num_imputed.drop(columns_to_drop, axis=1)

    return df_num_imputed


def remove_atypical_values(df):
    atypical_idx = set()

    n_times = 7

    for column in df.select_dtypes(include=['int64', 'float64']):
        # Calcular la media y la desviación estándar de la columna actual
        mean = df[column].mean()
        std_deviation = df[column].std()

        # Definir los límites superior e inferior para la columna actual
        lower_bound = mean - n_times * std_deviation
        upper_bound = mean + n_times * std_deviation

        # Identificar los valores atípicos en la columna actual
        atypical_idx_df = df[(df[column] < lower_bound) |
                          (df[column] > upper_bound)]

        # Imprimir los valores atípicos para la columna actual
        atypical_idx.update(atypical_idx_df.index)

    df = df.drop(list(atypical_idx))
    print("{total} atypical values have been removed".format(
        total=len(atypical_idx)))

    return df


def label_encode(df):
    # Seleccionamos las variables categóricas (sólo aplicaremos el one-hot sobre ellas)
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()

    le = LabelEncoder()
    df[cat_cols] = df[cat_cols].apply(le.fit_transform)

    return df


def split_and_normalize(df, test_size):
    X_train, X_test = train_test_split(df, test_size=test_size)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_test

def preprocess_data_server_side(df):
    string_cols = df.select_dtypes(include=['object']).columns

    # Convert columns of type string to type category
    df[string_cols] = df[string_cols].astype('category')   

    # Remove vars whose null percentage exceed threshold
    df = remove_high_null_vars(df)

    # Remove samples whose null percentage exceed threshold
    df = remove_high_null_samples(df)

    # Imput NaN values in both categorical and numerical variables
    df_num_imputed, df_cat_imputed = imput_nan_values(df)

    # Remove correlated numerical variables
    df_num_imputed = remove_correlated_vars(df_num_imputed)

    # Merge the two dataframe versions in one
    df = pd.concat([df_num_imputed, df_cat_imputed], ignore_index=True, axis=1)
    df.columns = df_num_imputed.columns.tolist() + df_cat_imputed.columns.tolist()

    string_cols = df.select_dtypes(include=['object']).columns

    # Convert columns of type string to type category
    df[string_cols] = df[string_cols].astype('category')


    return df


def preprocess_data_client_side(df, test_size):
    string_cols = df.select_dtypes(include=['object']).columns

    # Convert columns of type string to type category
    df[string_cols] = df[string_cols].astype('category')   

    # Remove samples whose null percentage exceed threshold
    df = remove_high_null_samples(df)

    # Imput NaN values in both categorical and numerical variables
    df_num_imputed, df_cat_imputed = imput_nan_values(df)

    # Merge the two dataframe versions in one
    df = pd.concat([df_num_imputed, df_cat_imputed], ignore_index=True, axis=1)
    df.columns = df_num_imputed.columns.tolist() + df_cat_imputed.columns.tolist()
    string_cols = df.select_dtypes(include=['object']).columns

    # Convert columns of type string to type category
    df[string_cols] = df[string_cols].astype('category')

    # Remove atypical values
    df = remove_atypical_values(df)

    # Apply label encoding over categorical variables
    df = label_encode(df)

    # Split dataframe in train/test and normalize values
    X_train, X_test = split_and_normalize(df, test_size)

    return X_train, X_test, len(df.columns)

def preprocess_data_centralized(df, test_size):
    string_cols = df.select_dtypes(include=['object']).columns

    # Convert columns of type string to type category
    df[string_cols] = df[string_cols].astype('category')

    # Remove vars whose null percentage exceed threshold
    df = remove_high_null_vars(df)

    # Remove samples whose null percentage exceed threshold
    df = remove_high_null_samples(df)
    # Imput NaN values in both categorical and numerical variables
    df_num_imputed, df_cat_imputed = imput_nan_values(df)

    # Remove correlated numerical variables
    df_num_imputed = remove_correlated_vars(df_num_imputed)

    # Merge the two dataframe versions in one
    df = pd.concat([df_num_imputed, df_cat_imputed], ignore_index=True, axis=1)
    df.columns = df_num_imputed.columns.tolist() + df_cat_imputed.columns.tolist()
    
    string_cols = df.select_dtypes(include=['object']).columns

    # Convert columns of type string to type category
    df[string_cols] = df[string_cols].astype('category')

    # Remove atypical values
    df = remove_atypical_values(df)

    # Apply label encoding over categorical variables
    df = label_encode(df)

    # Split dataframe in train/test and normalize values
    X_train, X_test = split_and_normalize(df, test_size)

    return X_train, X_test, len(df.columns)




