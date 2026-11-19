# Libraries
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

# Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso
#from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

#from DiamondPricePrediction import GradientBoostingRegressorModel

st.set_page_config(
    page_title="Diamond Price Prediction",
    layout="wide",  # Options: "centered" (default) or "wide"
    initial_sidebar_state="expanded",
)

st.title("Diamond Price Prediction")

# Data Collection
Data = pd.read_csv('diamonds.csv')

EDANumCols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
EDACatCols = ['cut', 'color', 'clarity']

CBOpenDataset = st.sidebar.checkbox("Open Dataset")
if CBOpenDataset:
    st.write(Data)

# EDA
Cl1, Cl2 = st.columns(2)
CBEDA = st.sidebar.checkbox("EDA")
if CBEDA:
    SBTEDA = st.sidebar.selectbox('Type of EDA', ['Univariate', 'Bivariate', 'Multivariate'])
    if SBTEDA == 'Univariate':
        ColName = st.sidebar.selectbox('Select Column', Data.columns)
        if ColName in EDANumCols:
            Cl1.plotly_chart(px.histogram(Data[ColName]))
            Cl2.plotly_chart(px.box(Data[ColName]))
        elif ColName in EDACatCols:
            DataCatEDA = Data.groupby(ColName)['price'].count().reset_index()
            DataCatEDA = DataCatEDA.rename(columns={'price': 'Count'}, inplace = True)

            Cl1.plotly_chart(px.bar(DataCatEDA, ColName, 'Count'))
            Cl2.plotly_chart(px.pie(DataCatEDA, ColName, 'Count'))
        else:
            pass
    elif SBTEDA == 'Bivariate':
        ColName = st.sidebar.selectbox('Select Column', Data.columns)
        TargetCol = Data['price']
        if ColName in EDANumCols:
            Cl1.plotly_chart(px.scatter(Data, ColName, 'price'))

            DataNumEDA = Data.groupby(ColName)['price'].mean().reset_index()
            Cl2.plotly_chart(px.line(DataNumEDA, ColName, 'price'))
    else:
        pass


#st.write(Data.head())
#st.write(Data.shape)
#st.write(Data.info())
#st.write(Data.describe())
#st.write(Data.columns)

# Train Test Split

X = Data.drop(columns = 'price')
Y = Data['price']
#st.write(X.shape, Y.shape)

TestSize = st.sidebar.slider("Select Test Size", 0.10, 0.95, 0.20, 0.05)

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = TestSize, random_state = 5)

CBShape = st.sidebar.checkbox("Check Shape")
if CBShape:
    st.write(f'X Train{XTrain.shape}')
    st.write(f'X Test{XTest.shape}')
    st.write(f'Y Train{YTrain.shape}')
    st.write(f'Y Test{YTest.shape}')

# Separate Numerical and Categorical XTrain, XTest

NumXTrain = XTrain.drop(columns=['cut', 'color', 'clarity'])
NumXTest = XTest.drop(columns=['cut', 'color', 'clarity'])

CatXTrain = XTrain[['cut', 'color', 'clarity']]
CatXTest = XTest[['cut', 'color', 'clarity']]

# Scaling

ScalerSS = StandardScaler()

NumXTrainS = ScalerSS.fit_transform(NumXTrain)
NumXTrainS = pd.DataFrame(NumXTrainS, columns = NumXTrain.columns, index = NumXTrain.index)

NumXTestS = ScalerSS.transform(NumXTest)
NumXTestS = pd.DataFrame(NumXTestS, columns = NumXTest.columns, index = NumXTest.index)

# Encoding

EncoderOE = OrdinalEncoder(categories = [['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                                       ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                                       ['I1', 'SI2', 'SI1','VS2','VS1','VVS2', 'VVS1', 'IF']])

CatXTrainE = EncoderOE.fit_transform(CatXTrain)
CatXTrainE = pd.DataFrame(CatXTrainE, columns = CatXTrain.columns, index = CatXTrain.index)

CatXTestE = EncoderOE.transform(CatXTest)
CatXTestE = pd.DataFrame(CatXTestE, columns = CatXTest.columns, index = CatXTest.index)

# join Numerical and Categorical NumXTrainS + CatXTrainE, NumXTestS + CatXTestE

NewXTrain = NumXTrainS.join(CatXTrainE)

NewXTest = NumXTestS.join(CatXTestE)

# Models Prediction

Models = ['Linear Regression', 'Lasso', 'Ridge', 'Decision Tree Regressor',
          'Random Forest Regressor', 'Gradient Boosting Regressor', 'Ada Boost Regressor',
          'SVR', 'KNeighbors Regressor', 'Polynomial Features']

ModelSelection = st.sidebar.selectbox('Select Model', Models)
if ModelSelection == Models[0]:
    LinearRegressionModel = LinearRegression()
    LinearRegressionModelResult = LinearRegressionModel.fit(NewXTrain, YTrain)

    CBShow = st.checkbox('Show Result')
    if CBShow:
        Prediction = LinearRegressionModel.predict(NewXTest)
        MSE = mean_squared_error(YTest, Prediction)
        R2 = r2_score(YTest, Prediction)
        st.write(f'{Models[0]} is ')
        st.write(f'MSE : {MSE}')
        st.write(f'R2 : {R2}')

        Models = {'Predicted_model': LinearRegressionModelResult, "Scaler": ScalerSS, "Encoder": EncoderOE}
        # Convert dictionary into pickle bytes
        ModelBytes = pickle.dumps(Models)

        # Download button
        st.download_button(
            label="Download Models",
            data=ModelBytes,
            file_name="LinearRegressionModel.pkl")

elif ModelSelection == Models[1]:
    LassoModel = Lasso(alpha=0.5)  # 0-1
    LassoModelResult = LassoModel.fit(NewXTrain, YTrain)

    CBShow = st.checkbox('Show Result')
    if CBShow:
        Prediction = LassoModel.predict(NewXTest)
        MSE = mean_squared_error(YTest, Prediction)
        R2 = r2_score(YTest, Prediction)
        st.write(f'{Models[1]} is ')
        st.write(f'MSE : {MSE}')
        st.write(f'R2 : {R2}')

        Models = {'Predicted_model': LassoModelResult, "Scaler": ScalerSS, "Encoder": EncoderOE}
        # Convert dictionary into pickle bytes
        ModelBytes = pickle.dumps(Models)

        # Download button
        st.download_button(
            label="Download Models",
            data=ModelBytes,
            file_name="LassoModel.pkl")

elif ModelSelection == Models[2]:
    RidgeModel = Ridge(alpha=2.0)
    RidgeModelResult = RidgeModel.fit(NewXTrain, YTrain)

    CBShow = st.checkbox('Show Result')
    if CBShow:
        Prediction = RidgeModel.predict(NewXTest)
        MSE = mean_squared_error(YTest, Prediction)
        R2 = r2_score(YTest, Prediction)
        st.write(f'{Models[2]} is ')
        st.write(f'MSE : {MSE}')
        st.write(f'R2 : {R2}')

        Models = {'Predicted_model': RidgeModelResult, "Scaler": ScalerSS, "Encoder": EncoderOE}
        # Convert dictionary into pickle bytes
        ModelBytes = pickle.dumps(Models)

        # Download button
        st.download_button(
            label="Download Models",
            data=ModelBytes,
            file_name="RidgeModel.pkl")

elif ModelSelection == Models[3]:
    DecisionTreeRegressorModel = DecisionTreeRegressor()
    DecisionTreeRegressorModelResult = DecisionTreeRegressorModel.fit(NewXTrain, YTrain)

    CBShow = st.checkbox('Show Result')
    if CBShow:
        Prediction = DecisionTreeRegressorModel.predict(NewXTest)
        MSE = mean_squared_error(YTest, Prediction)
        R2 = r2_score(YTest, Prediction)
        st.write(f'{Models[3]} is ')
        st.write(f'MSE : {MSE}')
        st.write(f'R2 : {R2}')

        Models = {'Predicted_model': DecisionTreeRegressorModelResult, "Scaler": ScalerSS, "Encoder": EncoderOE}
        # Convert dictionary into pickle bytes
        ModelBytes = pickle.dumps(Models)

        # Download button
        st.download_button(
            label="Download Models",
            data=ModelBytes,
            file_name="DecisionTreeRegressorModel.pkl")

elif ModelSelection == Models[4]:
    RandomForestRegressorModel = RandomForestRegressor()
    RandomForestRegressorModelResult = RandomForestRegressorModel.fit(NewXTrain, YTrain)

    CBShow = st.checkbox('Show Result')
    if CBShow:
        Prediction = RandomForestRegressorModel.predict(NewXTest)
        MSE = mean_squared_error(YTest, Prediction)
        R2 = r2_score(YTest, Prediction)
        st.write(f'{Models[4]} is ')
        st.write(f'MSE : {MSE}')
        st.write(f'R2 : {R2}')

        Models = {'Predicted_model': RandomForestRegressorModelResult, "Scaler": ScalerSS, "Encoder": EncoderOE}
        # Convert dictionary into pickle bytes
        ModelBytes = pickle.dumps(Models)

        # Download button
        st.download_button(
            label="Download Models",
            data=ModelBytes,
            file_name="RandomForestRegressorModel.pkl")

elif ModelSelection == Models[5]:
    GradientBoostingRegressorModel = GradientBoostingRegressor()
    GradientBoostingRegressorModelResult = GradientBoostingRegressorModel.fit(NewXTrain, YTrain)

    CBShow = st.checkbox('Show Result')
    if CBShow:
        Prediction = GradientBoostingRegressorModel.predict(NewXTest)
        MSE = mean_squared_error(YTest, Prediction)
        R2 = r2_score(YTest, Prediction)
        st.write(f'{Models[5]} is ')
        st.write(f'MSE : {MSE}')
        st.write(f'R2 : {R2}')

        Models = {'Predicted_model': GradientBoostingRegressorModelResult, "Scaler": ScalerSS, "Encoder": EncoderOE}
        # Convert dictionary into pickle bytes
        ModelBytes = pickle.dumps(Models)

        # Download button
        st.download_button(
            label="Download Models",
            data=ModelBytes,
            file_name="GradientBoostingRegressorModel.pkl")

elif ModelSelection == Models[6]:
    AdaBoostRegressorModel = AdaBoostRegressor()
    AdaBoostRegressorModelResult = AdaBoostRegressorModel.fit(NewXTrain, YTrain)

    CBShow = st.checkbox('Show Result')
    if CBShow:
        Prediction = AdaBoostRegressorModel.predict(NewXTest)
        MSE = mean_squared_error(YTest, Prediction)
        R2 = r2_score(YTest, Prediction)
        st.write(f'{Models[6]} is ')
        st.write(f'MSE : {MSE}')
        st.write(f'R2 : {R2}')

        Models = {'Predicted_model': AdaBoostRegressorModelResult, "Scaler": ScalerSS, "Encoder": EncoderOE}
        # Convert dictionary into pickle bytes
        ModelBytes = pickle.dumps(Models)

        # Download button
        st.download_button(
            label="Download Models",
            data=ModelBytes,
            file_name="AdaBoostRegressorModel.pkl")

elif ModelSelection == Models[7]:
    SVRModel = SVR()
    SVRModelResult = SVRModel.fit(NewXTrain, YTrain)

    CBShow = st.checkbox('Show Result')
    if CBShow:
        Prediction = SVRModel.predict(NewXTest)
        MSE = mean_squared_error(YTest, Prediction)
        R2 = r2_score(YTest, Prediction)
        st.write(f'{Models[7]} is ')
        st.write(f'MSE : {MSE}')
        st.write(f'R2 : {R2}')

        Models = {'Predicted_model': SVRModelResult, "Scaler": ScalerSS, "Encoder": EncoderOE}
        # Convert dictionary into pickle bytes
        ModelBytes = pickle.dumps(Models)

        # Download button
        st.download_button(
            label="Download Models",
            data=ModelBytes,
            file_name="SVRModel.pkl")

elif ModelSelection == Models[8]:
    KNeighborsRegressorModel = KNeighborsRegressor()
    KNeighborsRegressorModelResult = KNeighborsRegressorModel.fit(NewXTrain, YTrain)

    CBShow = st.checkbox('Show Result')
    if CBShow:
        Prediction = KNeighborsRegressorModel.predict(NewXTest)
        MSE = mean_squared_error(YTest, Prediction)
        R2 = r2_score(YTest, Prediction)
        st.write(f'{Models[8]} is ')
        st.write(f'MSE : {MSE}')
        st.write(f'R2 : {R2}')

        Models = {'Predicted_model': KNeighborsRegressorModelResult, "Scaler": ScalerSS, "Encoder": EncoderOE}
        # Convert dictionary into pickle bytes
        ModelBytes = pickle.dumps(Models)

        # Download button
        st.download_button(
            label="Download Models",
            data=ModelBytes,
            file_name="KNeighborsRegressorModel.pkl")

else:
    degree = 2
    PolynomialFeaturesModel = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    PolynomialFeaturesModelResult = PolynomialFeaturesModel.fit(NewXTrain, YTrain)

    CBShow = st.checkbox('Show Result')
    if CBShow:
        Prediction = PolynomialFeaturesModel.predict(NewXTest)
        MSE = mean_squared_error(YTest, Prediction)
        R2 = r2_score(YTest, Prediction)
        st.write(f'{Models[9]} is ')
        st.write(f'MSE : {MSE}')
        st.write(f'R2 : {R2}')

        Models = {'Predicted_model': PolynomialFeaturesModelResult, "Scaler": ScalerSS, "Encoder": EncoderOE}
        # Convert dictionary into pickle bytes
        ModelBytes = pickle.dumps(Models)

        # Download button
        st.download_button(
            label="Download Models",
            data=ModelBytes,
            file_name="PolynomialFeaturesModel.pkl")
