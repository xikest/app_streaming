##### 기본 정보 입력 #####
import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import pos_tag
from wordcloud import WordCloud
import plotly.express as px

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def make_dataframe_ex() -> pd.DataFrame:
    # 가상의 데이터 생성
    comments = {
        'comments': [
            "This is a sample comment about data analysis. Data analysis is a crucial step in any research or business decision-making process. It involves collecting, cleaning, and interpreting data to gain valuable insights. Data analysts use various tools and techniques to uncover patterns and trends in data. In today's data-driven world, data analysis skills are in high demand.",
            "Natural language processing (NLP) is a fascinating field of study. NLP focuses on the interaction between computers and human language. NLP applications include sentiment analysis, machine translation, chatbots, and more. NLP researchers develop algorithms to understand and generate human language. The possibilities in NLP seem endless, and it's an exciting area to explore.",
            "Machine learning is revolutionizing industries across the globe. It's the science of getting computers to learn and act like humans do. Machine learning algorithms are used in recommendation systems, image recognition, autonomous vehicles, and many other areas. As machine learning advances, it continues to shape the future of technology and innovation.",
            "Python is a versatile programming language commonly used in data science and machine learning. Its readability and extensive libraries make it a popular choice among data scientists. Python's simplicity and flexibility make it an excellent language for analyzing and visualizing data. It's no wonder that Python is a go-to language for data professionals.",
            "Artificial intelligence (AI) is a transformative technology with applications in healthcare, finance, and more. AI systems can perform tasks that typically require human intelligence. These systems learn from data, recognize patterns, and make decisions. The growth of AI is expected to drive significant changes in various industries."
        ]
    }
    # 데이터프레임 생성
    df = pd.DataFrame(comments)
    return df


def main():
    # 기본 설정
    st.set_page_config(
        page_title="plot stream",
        layout="wide")

    # session state 초기화
    st.session_state.setdefault("tab1", None)
    st.session_state.setdefault("tab2", None)
    st.session_state.setdefault("tab3", None)

    
    # 제목
    st.header("Visualization Streaming")
    # 구분선
    #st.markdown("---")

    # 사이드바 생성
    with st.sidebar:
        # with st.form(key='my_form'):
        #     username = st.text_input('Username')
        #     password = st.text_input('Password')
        #     st.form_submit_button('Login')

        # 기본 설명
        with st.expander("Project Description", expanded=True):
            st.write(
                """     
                - This project supports basic text analysis.
                """
            )
        st.markdown("---")
        st.markdown("이 프로젝트가 도움이 되었다면,")
        st.markdown("커피 한 잔은 큰 격려가 됩니다. ☕️")
        st.markdown("---")
        st.write(
            """     
            Written by TJ.Kim
            """
        )

    # Insert containers separated into tabs:
    tab1, tab2, tab3 = st.tabs(["Word Frequency", "Correlation", "LDA"])
    # tab1.write("EDA")
    # tab2.write("plot2")
    # tab3.write("plot3")


    with tab1:

        # 기능 구현 공간
        col1_tab1, col2_tab1 = st.columns(2)
        with col1_tab1:
            flag_word_freq_df = False
            # 오른쪽 영역 작성
            st.subheader("1. Data Preparation")
            df_example = make_dataframe_ex()
            # st.info('Input Data Form Example', icon="ℹ️")
            st.write("▶ Input Data Form Example")
            st.dataframe(df_example.head(2))

            data_uploaded = st.file_uploader("▶ Upload CSV or Excel files only.")
            if data_uploaded is not None:
                if data_uploaded.name.endswith('.csv'):
                    df = pd.read_csv(data_uploaded)
                    # st.success('Data read success!', icon="✅")
                elif data_uploaded.name.endswith('.xlsx'):
                    df = pd.read_excel(data_uploaded, engine='openpyxl')
                    # st.success('Data read success!', icon="✅")

                else:
                    st.error("This file format is not supported. Please upload a CSV or Excel file.")
                    st.stop()

                st.subheader("2. Data Preview")
                st.write("▶ Part of the data read")
                # st.info('Part of the data read', icon="ℹ️")
                st.dataframe(df.head(3))

                # 데이터 처리
                try:
                    try:
                        comments = df['comments']
                    except KeyError:
                        comments = df.iloc[:, 0]

                    all_words = []
                    for comment in comments:
                        tokens = word_tokenize(comment)  # 문장을 단어로 토큰화
                        all_words.extend(tokens)

                    # 불용어 제거
                    stop_words = set(stopwords.words('english'))
                    filtered_words = [word.lower() for word in all_words if word.isalnum() and word.lower() not in stop_words]

                    # 명사만 추출
                    nouns = [word for (word, tag) in pos_tag(filtered_words) if tag.startswith('N')]
                    # 명사 빈도를 계산
                    noun_counts = FreqDist(nouns)
                    # 데이터프레임 생성
                    df_word_freq = pd.DataFrame(list(noun_counts.items()), columns=['Nouns', 'Frequency'])
                    # 빈도순으로 정렬
                    df_word_freq = df_word_freq.sort_values(by='Frequency', ascending=False)
                    st.subheader("3. Analysis results")


                    st.write("▶ Partial analysis results")
                    # st.info('Partial analysis results.', icon="ℹ️")
                    st.dataframe(df_word_freq.head(3))
                    st.session_state["tab1"] = {"df_word_freq": df_word_freq, "nouns": nouns}

                    # 다운로드 버튼 추가
                    st.write("▶ Download Analysis Results")
                    csv_word_freq = df_word_freq.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Press to Download",
                        csv_word_freq,
                        "word_freq_analysis.csv",
                        "text/csv",
                        key='download-csv'
                    )


                except:
                    st.error('Please verify the file format', icon="🚨")
                    # st.subheader("3. Please verify the file format")
        
        st.write("▶  Visualization")        
        with col2_tab1:
            if st.session_state["tab1"] is not None:
            # 오른쪽 영역 작성
                tab1_col2_tab1, tab2_col2_tab1 = st.tabs(["Plot", "Word Cloud"])   
                with tab1_col2_tab1:
                    # st.subheader("Plot")
                    df = st.session_state["tab1"]["df_word_freq"]
                    top_words = df.head(10)
                    fig = px.bar(top_words, x='Nouns', y='Frequency', title="Top 10 Words Frequency")
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(width=330, height=330)
                    st.plotly_chart(fig)
                with tab2_col2_tab1:
                    # st.subheader("Word Cloud")
                    nouns = st.session_state["tab1"]["nouns"]
                     # Word Cloud 생성 800*400
                    wordcloud = WordCloud(width=400, height=400, background_color="white").generate(" ".join(nouns))

                    # Word Cloud를 Plotly 그래프로 표시
                    fig = px.imshow(wordcloud, binary_string=True)
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    fig.update_layout(width=330, height=330)
                    st.plotly_chart(fig)
    # 두 번째 탭: Correlation Plot
    with tab2:
        st.subheader("Correlation Plot Content")
    # 세 번째 탭: LDA
    with tab3:
        st.subheader("LDA Content")

if __name__ == "__main__":
    main()
