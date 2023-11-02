import pandas as pd
from nltk.probability import FreqDist
from nltk import pos_tag
from wordcloud import WordCloud
import plotly.express as px
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim import corpora
import re
import nltk

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
    # instruction
    st.markdown("**※ Upload CSV or Excel files only.**")
    st.write("(1) 'comments' column is the subject of analysis. Use the column name 'comments.'")
    st.write("(2) If no column name is specified, the first column will be the subject of analysis.")
    return st.dataframe(df.head(2))
def read_comments_from(data_uploaded, column_name="comments")->pd.DataFrame:
    df = pd.DataFrame()
    if data_uploaded.name.endswith('.csv'):
        df = pd.read_csv(data_uploaded)
    elif data_uploaded.name.endswith('.xlsx'):
        df = pd.read_excel(data_uploaded, engine='openpyxl')
    else:
        st.error("This file format is not supported. Please upload a CSV or Excel file.")
        st.stop()
    try:
        # st.dataframe(df.head(3))
        comments = df.loc[:, column_name]
    except KeyError:
        comments = df.iloc[:, 0]
    return comments
def prepare_networkg(text) ->"corpus, dictionary":

    # 텍스트 전처리: 소문자로 변환, 특수 문자 및 숫자 제거
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 단어 토큰화
    words = word_tokenize(text)
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # LDA를 위한 토픽 모델링 준비
    # 형용사와 명사 추출 (품사 태깅을 활용)
    nouns_adjectives = [word for word, pos in nltk.pos_tag(filtered_words) if
                        pos in ['NN', 'NNS', 'JJ']]

    # LDA 모델을 위한 말뭉치 생성
    dictionary = corpora.Dictionary([nouns_adjectives])
    corpus = [dictionary.doc2bow(nouns_adjectives)]
    return corpus, dictionary
def prepare_nouns(comments):
    all_words = []
    # nltk data download
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

    for comment in comments:
        tokens = word_tokenize(comment)  # tokenize
        all_words.extend(tokens)
    # stopward
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in all_words if
                      word.isalnum() and word.lower() not in stop_words]
    # nouns
    nouns = [word for (word, tag) in pos_tag(filtered_words) if tag.startswith('N')]
    return nouns
def prepare_word_freq(nouns) -> pd.DataFrame:
    # nouns frequncy
    noun_counts = FreqDist(nouns)
    df_word_freq = pd.DataFrame(list(noun_counts.items()), columns=['Nouns', 'Frequency'])
    # sorted
    df_word_freq = df_word_freq.sort_values(by='Frequency', ascending=False)
    return df_word_freq
def download_df_as_csv(df):
    csv_word_freq = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download",
        csv_word_freq,
        "word_freq_analysis.csv",
        "text/csv",
        key='download-csv'
    )
    return st.dataframe(df.head(3))
def plot_freq(df_word_freq, num_dis:int=5):
    top_words = df_word_freq.head(num_dis)
    fig = px.bar(top_words, x='Nouns', y='Frequency', title="Top 10 Words Frequency")
    fig.update_xaxes(tickangle=45)
    fig.update_layout(width=800, height=500)
    return st.plotly_chart(fig, use_container_width=True)
def plot_wordcloud(nouns):
    # Word Cloud: 800*400
    wordcloud = WordCloud(width=800, height=500, background_color="white").generate(" ".join(nouns))
    fig = px.imshow(wordcloud, binary_string=True)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # fig.update_layout(width=330, height=330)
    return st.plotly_chart(fig, use_container_width=True)
def plot_networkg(corpus, dictionary):
    # LDA 모델 학습
    lda_model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

    # LDA 모델에서 주제 추출
    topics = lda_model.show_topics(num_topics=3, num_words=5)  # 주제당 상위 5개 단어 출력

    # 그래프 생성
    G = nx.Graph()

    # 주제를 노드로 추가
    for topic, words in topics:
        node_label = f"Topic {topic}"
        G.add_node(node_label, node_type='topic')  # 토픽 노드에 'node_type' 속성 추가
        word_list = words.split('+')
        for word in word_list:
            prob, word = word.split('*')
            word = word.strip()
            G.add_node(word, node_type='word')  # 단어 노드에 'node_type' 속성 추가
            G.add_edge(node_label, word, weight=float(prob))

    # 노드 크기를 최소 5에서 최대 50으로 정규화하여 설정
    node_degrees = dict(G.degree)
    min_size = 100
    max_size = 1000
    node_size = [
        min_size + (max_size - min_size) * (node_degrees[node] - min(node_degrees.values())) / (
                max(node_degrees.values()) - min(node_degrees.values()) + 1) for node in G.nodes]

    # graph viusalize
    pos = nx.spring_layout(G, seed=42)

    edge_width = [data['weight'] * 5 for _, _, data in G.edges(data=True)]

    node_colors = ['lightblue' if G.nodes[node]['node_type'] == 'topic' else 'lightgray' for node in
                   G.nodes]  # 토픽 노드와 단어 노드에 다른 색상 적용

    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_size=node_size, width=edge_width, node_color=node_colors,
            font_size=8)
    plt.axis('off')
    return st.pyplot(plt, use_container_width=True)


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

            data_uploaded = st.file_uploader("")
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
                    plot_freq(df_word_freq, num_dis=5)
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
