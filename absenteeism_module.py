import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Scaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

# Absenteeism model class
class absenteeism_model():
    def __init__(self, model_file, scaler_file):
        with open(model_file, 'rb') as model_f, open(scaler_file, 'rb') as scaler_f:
            self.reg = pickle.load(model_f)
            self.scaler = pickle.load(scaler_f)
            self.data = None

    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file, delimiter=',')
        self.df_with_predictions = df.copy()
        df = df.drop(['ID'], axis=1)
        df['Absenteeism Time in Hours'] = 'NaN'

        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
        df = df.drop(['Reason for Absence'], axis=1)
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                        'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                        'Pet', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
        df.columns = column_names
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense',
                                  'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                                  'Children', 'Pet', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        list_months = [df['Date'][i].month for i in range(df.shape[0])]
        df['Month Value'] = list_months
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())
        df = df.drop(['Date'], axis=1)
        column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                            'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average',
                            'Body Mass Index', 'Education', 'Children', 'Pet', 'Absenteeism Time in Hours']
        df = df[column_names_upd]
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        df = df.fillna(value=0)
        df = df.drop(['Absenteeism Time in Hours'], axis=1)
        df = df.drop(['Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis=1)
        self.preprocessed_data = df.copy()
        self.data = self.scaler.transform(df)

    def predicted_probability(self):
        if self.data is not None:
            return self.reg.predict_proba(self.data)[:, 1]

    def predicted_output_category(self):
        if self.data is not None:
            return self.reg.predict(self.data)

    def predicted_outputs(self):
        if self.data is not None:
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data

# Streamlit interface
st.title("Absenteeism Prediction")
st.write("Upload your CSV file to predict absenteeism probabilities and categories.")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Instantiate the model
    model = absenteeism_model(model_file='/path/to/model', scaler_file='/path/to/scaler')
    model.load_and_clean_data(uploaded_file)
    
    # Display predictions
    st.write("Predicted Probabilities and Categories:")
    predictions = model.predicted_outputs()
    st.write(predictions)
