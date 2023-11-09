import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from functions.sentimentmanager import SentimentManager

def main():
    # basic setting
    st.set_page_config(
        page_title="plot stream",
        layout="wide")

    # session state initialize
    st.session_state["API_KEY"] = None
    st.session_state["result"] = None
    st.session_state["keywords"] = ["brightness", "color", "contrast", "reflection", "viewing"]

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


        st.session_state["API_KEY"] = API_KEY
        stm = SentimentManager(API_KEY)
        # print(st.session_state["API_KEY"] is True)
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
        df_sample_sentences = stm.sample_sentences()
        stm.download_df_as_csv(df_sample_sentences, file_name="sample_text_data", key="download_text_sample_csv", label="Sample download")
        if st.session_state["API_KEY"]:
            df_uploaded = st.file_uploader("Upload Text data", key="text_data")

            # text_data_uploaded = df_sample_sentences
            st.markdown("---")
            if df_uploaded:  ## 업로드
                df_uploaded = stm.read_df_from(df_uploaded)
                df_uploaded['sentences'] = df_uploaded['sentences'].apply(stm.preprocess_text)
                try:


                    df_sentences = df_uploaded
                    list_sentences = [sentence for sentence in df_sentences["sentences"]]  # 리스트로 변환
                    list_keywords = st.session_state["keywords"]
                    df_analyzed_results = stm.analyze_sentences(list_sentences, list_keywords)
                    st.subheader("2. Analysis results")
                    stm.download_df_as_csv(df_analyzed_results, file_name="sentiment_analysis", key="download_csv_text_analysis", label="Result download")
                    st.dataframe(df_analyzed_results.head(3))
                    st.session_state["result"] = df_analyzed_results
                    st.session_state["API_KEY"] = None
                    print('fin')
                except:
                    st.error('Re-Check', icon="🚨")
        else:
            st.error("PLEASE INPUT YOUR API-KEY")
    with col2:
        if  st.session_state["result"] is not None:
            st.subheader("3. Visualization")
            df_analyzed_results = st.session_state["result"]

            tab1, tab2, tab3, tab4, tab5 = st.tabs(st.session_state["keywords"])
            tabs = [tab1, tab2, tab3, tab4, tab5]

            columns = df_analyzed_results.columns
            for i, column in enumerate(columns):
                with tabs[i]:
                    sns.set_style("white")
                    fig, axes = plt.subplots(figsize=(10, 4))
                    sns.histplot(df_analyzed_results[column], kde=True, label=column, bins=10, ax=axes)
                    axes.set_ylabel("Density")
                    axes.set_title(f"{column}")
                    # axes.legend(loc="upper right")
                    axes.set_xlim(1, 10)
                    axes.set_xlabel("")
                    sns.despine()
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)


if __name__ == "__main__":
    main()

