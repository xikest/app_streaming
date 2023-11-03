# # # 단계 1: 데이터 로드 및 전처리
# # #
# # # load_data: 데이터를 로드하고 데이터프레임을 생성합니다.
# # # preprocess_data: 결측값을 보간하는 데이터 전처리를 수행합니다.
# # # 단계 2: 데이터 분석
# # #
# # # split_data: 수치형과 범주형 데이터로 데이터를 분리합니다.
# # # correlation_analysis: 수치형 데이터 간의 상관 관계를 시각화하여 분석합니다.
# # # missing_value_analysis: 결측값을 시각화하여 분석합니다.
# # # numerical_distribution_analysis: 수치형 데이터의 분포를 히스토그램으로 시각화하여 분석합니다.
# # # normality_analysis: 수치형 데이터의 정규성을 확인하기 위해 Q-Q 플롯을 시각화하여 분석합니다.
# # # categorical_distribution_analysis: 범주형 데이터의 분포를 바 차트로 시각화하여 분석합니다.
# # # 단계 3: PCA (주성분 분석)를 사용한 차원 축소
# # #
# # # pca_reduction: PCA를 사용하여 수치형 데이터의 차원을 축소하고 주성분을 추출합니다.
# # # 단계 4: t-SNE (t-Distributed Stochastic Neighbor Embedding)를 사용한 차원 축소
# # #
# # # tsne_reduction: t-SNE를 사용하여 수치형 데이터의 차원을 축소하고 데이터를 시각화합니다.
# # # 단계 5: MLP (다층 퍼셉트론) 모델 학습 및 평가
# # #
# # # mlp_model: MLP 회귀 모델을 생성하고 학습합니다.
# # # evaluate_mlp_model: MLP 모델을 평가하고 MSE (평균 제곱 오차)를 계산합니다.
# # # tsne_visualization: t-SNE로 축소된 데이터를 2차원 그래프로 시각화합니다.
# # # mlp_results_visualization: MLP 모델의 결과를 시각화하고 실제 값과 예측 값의 관계를 나타내며, MSE를 표시합니다.
# # # 단계 6: 실행 및 결과 확인
# # #
# # # main: 위의 단계를 실행하여 데이터 분석 및 모델링을 수행하고 결과를 확인합니다.
# # # 각 단계에서 주요 작업과 시각화를 수행하여 데이터 분석 및 모델링 프로세스를 완료합니다.
# #
# # colomn1은 전처리 및 데이터 시각화
# # column2은 예측 모델링 및 분류 모델
#
#
# #
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

#
def call_example_multi_numeric():
    data = {
        'target': [500, 550, 600, 480, 530, 520, 650, 690, 600, 550],
        'price': [100, 150, 200, 120, 180, 130, 210, 220, 250, 190],
        'customer': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
        'spec1': [1, 2, 3, 1, 2, 1, 3, 3, 2, 1],
        'spec2': ['Good', 'Excellent', 'Good', 'Excellent', 'Average', 'Good', 'Good', 'Excellent', 'Excellent',
                  'Average'],
        'spec3': ['North', 'South', 'North', 'West', 'South', 'North', 'West', 'North', 'South', 'West'],
        'spec4': [2020, 2021, 2022, 2020, 2021, 2020, 2022, 2022, 2021, 2020],

    }
    df = pd.DataFrame(data)
    st.markdown("**Supported Formats: CSV, Excel**")
    st.markdown("Excel (or CSV) Considerations: `target` column is assigned as the dependent variable.")
    return df
def read_numeric_from(data_uploaded, column_target: str = "target") -> pd.DataFrame:
    df = pd.DataFrame()
    supported_formats = ['.csv', '.xlsx', '.txt']
    if data_uploaded.name.endswith(tuple(supported_formats)):
        if data_uploaded.name.endswith('.csv'):
            df = pd.read_csv(data_uploaded)
        elif data_uploaded.name.endswith('.xlsx'):
            df = pd.read_excel(data_uploaded, engine='openpyxl')
    else:
        st.error("This file format is not supported. Please upload a CSV, Excel, or text file.")
        st.stop()
    try:
        y = df.loc[:, column_target]
        X = df.drop(column_target, axis=1)
    except KeyError:
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    # Check if y is not continuous numerical data
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("The 'y' column is not continuous numerical data.")
    df_combined = pd.concat([y, X], axis=1)
    return df_combined
def is_na(df):
    return df.isna().sum().mean() > 0
def preprocess_data(df):
    st.markdown("#### Preprocess")
    df.interpolate(method='linear', inplace=True)
    return df

def missing_value_analysis(df, title=""):
    st.write(f'- {title}')
    msno.matrix(df)
    st.pyplot(plt, use_container_width=True)
    return None



def split_data(df):
    y_column = df.columns[0]
    numerical_columns = df.select_dtypes(include=['number']).columns    # 숫자 데이터를 포함하는 열 선택
    categorical_columns = df.select_dtypes(exclude=['number']).columns   # 숫자 데이터를 포함하지 않는 열 선택
    return y_column, numerical_columns, categorical_columns


def one_hot_encode_categorical(df, categorical_columns):
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df_encoded


def plot_correlation(numerical_data, y) -> None:
    correlation_matrix = numerical_data.corrwith(y, method="spearman")
    correlation_matrix[y.name] = 1.0

    correlation_matrix = correlation_matrix.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    cax = ax.matshow(correlation_matrix.values.reshape(1, -1), cmap='Blues')

    plt.xticks(range(len(correlation_matrix)), correlation_matrix.index, rotation=45)
    plt.yticks([])  # y-axis 레이블을 표시하지 않음
    # plt.title('Correlation Map')
    st.pyplot(fig, use_container_width=True)
    return None

#
# def numerical_distribution_analysis(numerical_data):
#     histogram_fig = px.histogram(df, x=numerical_data.columns)
#     histogram_fig.update_layout(
#         title="Distribution of Numerical Features",
#         xaxis_title="Feature Value",
#         yaxis_title="Frequency"
#     )
#     st.plotly_chart(histogram_fig)
#
#
# def normality_analysis(numerical_data):
#     qq_plot = go.Figure()
#     for column in numerical_data.columns:
#         qq_data = numerical_data[column].dropna()
#         qq_plot.add_trace(go.Figure(data=stats.probplot(qq_data, plot=None)))
#     qq_plot.update_layout(
#         title="Q-Q Plot for Normality Check (Numerical Features)",
#         xaxis_title="Theoretical Quantiles",
#         yaxis_title="Sample Quantiles"
#     )
#     st.plotly_chart(qq_plot)
#
#
# def categorical_distribution_analysis(categorical_data):
#     for column in categorical_data.columns:
#         categorical_fig = px.bar(categorical_data, x=column, labels={column: 'Count'})
#         categorical_fig.update_layout(
#             title=f'Categorical Data Distribution ({column})',
#             xaxis_title=column,
#             yaxis_title="Count"
#         )
#         st.plotly_chart(categorical_fig)
#
#
# def pca_reduction(numerical_data):
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(numerical_data)
#     pca = PCA(n_components=2)
#     principal_components_pca = pca.fit_transform(scaled_data)
#     pca_df = pd.DataFrame(data=principal_components_pca, columns=['PC1', 'PC2'])
#     return pca_df
#
#
# def tsne_reduction(numerical_data):
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(numerical_data)
#     tsne = TSNE(n_components=2, random_state=42)
#     principal_components_tsne = tsne.fit_transform(scaled_data)
#     tsne_df = pd.DataFrame(data=principal_components_tsne, columns=['t-SNE1', 't-SNE2'])
#     return tsne_df
#
#
# def knn_reduction(numerical_data, k):
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(numerical_data)
#     knn = NearestNeighbors(n_neighbors=k)
#     knn.fit(scaled_data)
#     distances, indices = knn.kneighbors(scaled_data)
#     st.write("K-Nearest Neighbors (KNN) for Dimension Reduction:")
#     st.write("Distances to Nearest Neighbors:")
#     st.write(distances)
#     st.write("Indices of Nearest Neighbors:")
#     st.write(indices)
#
#
# def mlp_model(X_train, y_train):
#     mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
#     mlp_regressor.fit(X_train, y_train)
#     return mlp_regressor
#
#
# def decision_tree_model(X_train, y_train):
#     decision_tree = DecisionTreeRegressor(random_state=42)
#     decision_tree.fit(X_train, y_train)
#     return decision_tree
#
#
# def random_forest_model(X_train, y_train):
#     random_forest = RandomForestRegressor(random_state=42)
#     random_forest.fit(X_train, y_train)
#     return random_forest
#
#
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     return mse, y_pred
#
#
# def tsne_visualization(tsne_df):
#     fig = px.scatter(tsne_df, x='t-SNE1', y='t-SNE2',
#                      labels={'t-SNE1': 't-SNE Dimension 1', 't-SNE2': 't-SNE Dimension 2'})
#     fig.update_layout(
#         title="t-SNE Visualization",
#         xaxis_title="t-SNE Dimension 1",
#         yaxis_title="t-SNE Dimension 2"
#     )
#     st.plotly_chart(fig)
#
#
# def mlp_results_visualization(y_test, y_pred):
#     scatter_fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
#     scatter_fig.update_traces(marker=dict(size=5))
#     scatter_fig.update_layout(
#         title="Actual vs. Predicted Values",
#         xaxis_title="Actual",
#         yaxis_title="Predicted"
#     )
#
#     scatter_fig.add_shape(
#         type='line',
#         x0=y_test.min(),
#         y0=y_test.min(),
#         x1=y_test.max(),
#         y1=y_test.max(),
#         line=dict(color='red', dash='dash'),
#     )
#
#     st.plotly_chart(scatter_fig)
#
#
# def main():
#     df = load_data()
#
#     st.write("Step 1: Data Loading and Preprocessing")
#     df = preprocess_data(df)
#
#     st.write("Step 2: Data Analysis")
#     numerical_data, categorical_data = split_data(df)
#     correlation_analysis(numerical_data)
#     missing_value_analysis(df)
#     numerical_distribution_analysis(numerical_data)
#     normality_analysis(numerical_data)
#     categorical_distribution_analysis(categorical_data)
#
#     st.write("Step 3: Dimension Reduction (PCA)")
#     pca_df = pca_reduction(numerical_data)
#
#     st.write("Step 4: Dimension Reduction (t-SNE)")
#     tsne_df = tsne_reduction(numerical_data)
#     tsne_visualization(tsne_df)
#
#     st.write("Step 5: Dimension Reduction (K-Nearest Neighbors)")
#     k = 3  # Define the number of neighbors for KNN
#     knn_reduction(numerical_data, k)
#
#     st.write("Step 6: MLP Modeling")
#     X_train, X_test, y_train, y_test = train_test_split(pca_df, df['sales'], test_size=0.2, random_state=42)
#     mlp_regressor = mlp_model(X_train, y_train)
#     mlp_mse, mlp_y_pred = evaluate_model(mlp_regressor, X_test, y_test)
#     st.write("MLP Model MSE:", mlp_mse)
#
#     st.write("MLP Model Results Visualization:")
#     mlp_results_visualization(y_test, mlp_y_pred)
#
#     st.write("Step 7: Predictive Modeling with Decision Tree and Random Forest")
#     decision_tree = decision_tree_model(X_train, y_train)
#     random_forest = random_forest_model(X_train, y_train)
#
#     decision_tree_mse, decision_tree_y_pred = evaluate_model(decision_tree, X_test, y_test)
#     random_forest_mse, random_forest_y_pred = evaluate_model(random_forest, X_test, y_test)
#
#     st.write("Decision Tree Model MSE:", decision_tree_mse)
#     st.write("Random Forest Model MSE:", random_forest_mse)
#
#     st.write("Decision Tree Model Results Visualization:")
#     mlp_results_visualization(y_test, decision_tree_y_pred)
#
#     st.write("Random Forest Model Results Visualization:")
#     mlp_results_visualization(y_test, random_forest_y_pred)
#
#
# if __name__ == '__main__':
#     main()
