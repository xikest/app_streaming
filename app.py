
from functions import *
def main():
    # basic setting
    st.set_page_config(
        page_title="plot stream",
        layout="wide")

    # session state initialize
    st.session_state.setdefault("tab1", None)
    st.session_state.setdefault("tab2", None)
    st.session_state.setdefault("tab3", None)

    # Title
    st.header("Plot Visualization")

    # Side bar
    with st.sidebar:
        # Basic description
        with st.expander("Project Description", expanded=False):
            st.write(
                """     
                - This project supports basic text analysis.
                """
            )
        st.markdown("---")
        st.write("This project has been helpful, a cup of coffee would be a great encouragement. ☕️")
        st.markdown("---")
        st.write(
            """     
            Written by TJ.Kim
            """
        )

    # Insert containers separated into tabs:
    tab1, tab2, tab3 = st.tabs(["Text Analysis", "Numeric Analysis", "Time Series Analysis"])
    # tab1 = st.tabs(["Word Frequency"])
    # tab1.write("EDA")
    # tab2.write("plot2")
    # tab3.write("plot3")

    with tab1:
        # function
        col1_tab1, col2_tab1 = st.columns([1, 2])

        with col1_tab1:
            flag_word_freq_df = False
            # Right seg
            st.subheader("1. Data Preparation")
            call_example_comments()
            data_uploaded = st.file_uploader("※ Upload CSV, Excel, or Text files only.")
            if data_uploaded is not None:
                # st.subheader("Data Preview")
                # extract data
                try:
                    comments = read_comments_from(data_uploaded, column_name="comments")
                    comments_as_string = ' '.join(comments.astype(str))
                    nouns = prepare_nouns(comments)
                    df_word_freq = prepare_word_freq(nouns)
                    corpus, dictionary = prepare_networkg(comments_as_string)

                    st.session_state["tab1"] = {"plot_df_word_freq": df_word_freq,
                                                "wordcloud_nouns": nouns,
                                                "network_corpus":corpus,
                                                "network_dictionary":dictionary}
                    st.subheader("2. Analysis results")
                    # download btn
                    download_df_as_csv(df_word_freq)
                except:
                    st.error('Please verify the file format', icon="🚨")

        with col2_tab1:
            if st.session_state["tab1"] is not None:
                st.subheader("3. Visualization")
                tab1_col2_tab1, tab2_col2_tab1, tab3_col2_tab1 = st.tabs(["Plot", "Word Cloud", "Network Graph"])
                with tab1_col2_tab1:
                    df_word_freq = st.session_state["tab1"]["plot_df_word_freq"]
                    plot_freq(df_word_freq)
                with tab2_col2_tab1:
                    nouns = st.session_state["tab1"]["wordcloud_nouns"]
                    plot_wordcloud(nouns)
                with tab3_col2_tab1:
                    corpus = st.session_state["tab1"]["network_corpus"]
                    dictionary = st.session_state["tab1"]["network_dictionary"]
                    plot_networkg(corpus, dictionary)
    # second tab: Correlation Plot
    with tab2:
        st.subheader("In the conceptualization stage")
    # third tab: LDA
    with tab3:
        st.subheader("In the conceptualization stage")


if __name__ == "__main__":
    main()