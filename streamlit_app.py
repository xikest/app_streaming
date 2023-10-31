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


##### 메인 함수 #####
def main():
    # 기본 설정
    st.set_page_config(
        page_title="plot stream",
        layout="wide")

    flag_start = False

    # session state 초기화
    if "dataframe" not in st.session_state:
        st.session_state["dataframe"] = []

    # 제목
    st.header("plot streaming")
    # 구분선
    st.markdown("---")

    # 기본 설명
    with st.expander("plot streaming", expanded=True):
        st.write(
            """     
            - 이 프로젝트는 simple text 분석을 지원 합니다.
            """
        )

        st.markdown("")

    # 사이드바 생성
    with st.sidebar:
        with st.form(key='my_form'):
            username = st.text_input('Username')
            password = st.text_input('Password')
            st.form_submit_button('Login')

        st.markdown("---")
        st.markdown("만약 이 프로젝트가 도움이 되었다면, 커피 한 잔의 후원은 큰 격려가 됩니다. ☕️")
        st.markdown("---")
        st.write(
            """     
            Written by TJ.Kim
            """
        )

    # Insert containers separated into tabs:
    tab1, tab2, tab3 = st.tabs(["Word Frequency Visualization", "corr plot", "LDA"])
    tab1.write("plot1")
    tab2.write("plot2")
    tab3.write("plot3")
    # You can also use "with" notation:





    with tab1:

        # 기능 구현 공간
        col1, col2 = st.columns(2)
        with col1:
            flag_word_freq_df = False
            # 오른쪽 영역 작성
            st.subheader("데이터 준비")
            data_uploaded = st.file_uploader('File uploader')
            if data_uploaded is not None:
                if data_uploaded.name.endswith('.csv'):
                    df = pd.read_csv(data_uploaded)
                elif data_uploaded.name.endswith('.xlsx'):
                    df = pd.read_excel(data_uploaded)
                else:
                    st.error("지원하지 않는 파일 형식입니다. CSV 또는 Excel 파일을 업로드해 주세요.")
                    st.stop()

                st.subheader("Data Preview")
                st.dataframe(df)

                # 데이터 처리
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
                word_freq_df = pd.DataFrame(list(noun_counts.items()), columns=['Nouns', 'Frequency'])
                # 빈도순으로 정렬
                word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)
                st.dataframe(word_freq_df)
                st.session_state["tab1"] = {"word_freq_df":word_freq_df}
                st.session_state["tab1"] = {"nouns": nouns}

        with col2:
            # 오른쪽 영역 작성
            tab1, tab2 = st.tabs(["plot bar", "Word cloud"])
            tab1.write("plot bar")
            tab2.write("Word cloud")
            if st.session_state["tab1"] is not None:
                with tab1:
                    st.subheader("plot bar")
                    df = st.session_state["tab1"]["word_freq_df"]
                    top_words = df.head(10)
                    fig = px.bar(top_words, x='Nouns', y='Frequency', title="Top 10 Words Frequency")
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig)

                with tab2:
                    st.subheader("Word Cloud")
                    nouns = st.session_state["tab1"]["nouns"]
                     # Word Cloud 생성
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(nouns))

                    # Word Cloud를 Plotly 그래프로 표시
                    fig = px.imshow(wordcloud, binary_string=True)
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    st.plotly_chart(fig)

    # 두 번째 탭: Correlation Plot
    with tab2:
        st.subheader("Correlation Plot Content")

    # 세 번째 탭: LDA
    with tab3:
        st.subheader("LDA Content")

if __name__ == "__main__":
    main()