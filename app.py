from functions.sentimentmanager import sample_sentences
from functions.sentimentmanager import SentimentManager
from functions.sentimentmanager import plot_distribution

import pandas as pd
import streamlit as st

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


def main():
    # basic setting
    st.set_page_config(
        page_title="plot stream",
                       layout="wide")

    # session state initialize
    st.session_state.setdefault("tab1", None)
    
    # Title
    st.header("Plot Visualization")

    # Side bar
    with st.sidebar:
        # Basic description
        st.subheader("Project Description")
        st.write("This project supports basic analysis.")
        with st.expander("Usage", expanded=True):
            st.write("`sentiment Analysis`")
        st.markdown("---")
        # Open AI API 키 입력받기
        API_KEY = st.text_input(label="OPENAI API 키", placeholder="Enter Your API Key", value="",
                                       type="password")
        API_KEY = "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"
        st.session_state["API_KEY"] = API_KEY

        st.markdown("---")
        # GPT 모델을 선택하기 위한 라디오 버튼 생성
        model = st.radio(label="GPT 모델", options=["gpt-3.5-turbo", "gpt-4"])
        st.session_state["model"] = model

        st.markdown("---")
        st.write(
            """     
            Written by TJ.Kim ☕
            """
        )
        st.markdown("---")


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Data Preparation")
        df_sample_sentences = sample_sentences()
        download_df_as_csv(df_sample_sentences, file_name="sample_text_data", key="download_text_sample_csv", label="Sample download")
        st.dataframe(df_sample_sentences.head())
        text_data_uploaded = st.file_uploader("Upload Text data", key="text_data")
        # st.markdown("---")
        # download_df_as_csv(df_example_comments, file_name="sample_text_data", key="download_text_sample_csv", label="Sample")
        st.markdown("---")
        text_data_uploaded = df_sample_sentences
        if text_data_uploaded is not None:
            try:
                # sentences = read_comments_from(text_data_uploaded, column_name="sentences")
                sentences= df_sample_sentences
                # st.dataframe(sentences)
                sentences = [sentence for sentence in sentences["sentences"]] # 리스트로 변환
                keywords = ["color", "brightness"]
                sentimentManager = SentimentManager(API_KEY)

                dict_analyzed_results = sentimentManager.analyze_sentences(sentences, keywords)
                for key, value in dict_analyzed_results.items():
                    print(f"Sentence: '{key}' - Sentiment Scores: {value}")
                # 딕셔너리에서 칼럼 이름 추출



                st.subheader("2. Analysis results")
                df_results = pd.DataFrame(list(dict_analyzed_results.values()))
                st.session_state['result'] = df_results
                download_df_as_csv(df_results, file_name="sentiment_analysis", key="download_csv_text_analysis", label="Result download")
                st.dataframe(df_results.head(3))
            except:
                st.error('Please verify the file format', icon="🚨")
    with col2:
            st.subheader("3. Visualization")
            df_result_v = st.session_state.get("result")
           # df_result_v = st.session_state["result"]
            plot_distribution(df_result_v)

            #
            # tab1_col2_tab1, tab2_col2_tab1 = st.tabs(["Plot", "Word Cloud"])
            # with tab1_col2_tab1:
            #     df_result_v = st.session_state["result"]
            #     plot_distribution(df_result_v)
            # with tab2_col2_tab1:
            #     st.write("preparing")
            #     # nouns = st.session_state["result"]
            #     # plot_wordcloud(nouns)


if __name__ == "__main__":
    main()
