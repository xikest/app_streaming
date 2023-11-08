import pandas as pd
from wordcloud import WordCloud
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import re
import seaborn as sns


def call_example_comments() -> pd.DataFrame:
    # creating exmaple data
    comments = {
        'comments': [
            "This is a sample comment about data analysis. Data analysis is a crucial step in any research or business decision-making process. It involves collecting, cleaning, and interpreting data to gain valuable insights. Data analysts use various tools and techniques to uncover patterns and trends in data. In today's data-driven world, data analysis skills are in high demand.",
            "Natural language processing (NLP) is a fascinating field of study. NLP focuses on the interaction between computers and human language. NLP applications include sentiment analysis, machine translation, chatbots, and more. NLP researchers develop algorithms to understand and generate human language. The possibilities in NLP seem endless, and it's an exciting area to explore.",
            "Machine learning is revolutionizing industries across the globe. It's the science of getting computers to learn and act like humans do. Machine learning algorithms are used in recommendation systems, image recognition, autonomous vehicles, and many other areas. As machine learning advances, it continues to shape the future of technology and innovation.",
            "Python is a versatile programming language commonly used in data science and machine learning. Its readability and extensive libraries make it a popular choice among data scientists. Python's simplicity and flexibility make it an excellent language for analyzing and visualizing data. It's no wonder that Python is a go-to language for data professionals.",
            "Artificial intelligence (AI) is a transformative technology with applications in healthcare, finance, and more. AI systems can perform tasks that typically require human intelligence. These systems learn from data, recognize patterns, and make decisions. The growth of AI is expected to drive significant changes in various industries."
        ]
    }
    # dataframe
    df = pd.DataFrame(comments)
    st.markdown("**Supported Formats: CSV, Excel, Text**")
    st.markdown("Excel (or CSV) Considerations: `comments` column is the subject of analysis.")
    return df

def read_comments_from(data_uploaded, column_name="comments") -> pd.Series:
    df = pd.DataFrame()
    supported_formats = ['.csv', '.xlsx', '.txt']
    if data_uploaded.name.endswith(tuple(supported_formats)):
        if data_uploaded.name.endswith('.csv'):
            df = pd.read_csv(data_uploaded)
        elif data_uploaded.name.endswith('.xlsx'):
            df = pd.read_excel(data_uploaded, engine='openpyxl')
        elif data_uploaded.name.endswith('.txt'):
            df = pd.read_csv(data_uploaded, delimiter='\t')  # Assuming tab-separated text file
    else:
        st.error("This file format is not supported. Please upload a CSV, Excel, or text file.")
        st.stop()
    try:
        comments = df.loc[:, column_name]
    except KeyError:
        comments = df.iloc[:, 0]
    return comments



def download_df_as_csv(df: pd.DataFrame, file_name: str, key:str, preview=True, label:str="Download") -> None:

    csv_file = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label,
        csv_file,
        f"{file_name}.csv",
        "text/csv",
        key=key
    )
    # if preview:
    #     st.dataframe(df.head(3))
    return None


def plot_distribution(df):
    sns.set_style("white")
    num_columns = len(df.columns)
    num_rows = 1  # 각 행에 하나의 그래프를 배치
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(10, 4))

    for i, column in enumerate(df.columns):
        sns.histplot(df[column], kde=True, label=column, bins=5, ax=axes[i])
        axes[i].set_xlabel("Feature Value")
        axes[i].set_ylabel("Density")
        axes[i].set_title(f"Distribution of {column}")
        axes[i].legend()
        sns.despine()

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_wordcloud(nouns):
    # Create a WordCloud object with the desired settings
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(nouns))
    # Create a Matplotlib figure and axis
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # Display the Matplotlib figure within Streamlit
    st.pyplot(plt, use_container_width=True)

