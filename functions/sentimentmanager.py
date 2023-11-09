import pandas as pd
# from wordcloud import WordCloud
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from functions.aimanager import AIManager
from functions.logmanager import LogManager


def sample_sentences() -> pd.DataFrame:
    # creating exmaple data
    sentences = {
        'sentences': [
             "The product's hue and vividness are excellent.",
             "The brightness is great, but the color is not good.",
             "The viewing angle is disappointing, and the color is satisfactory."
     ]

    }
    # dataframe
    df = pd.DataFrame(sentences)
    st.markdown("**Supported Formats: CSV, Excel, Text**")
    st.markdown("Excel (or CSV) Considerations: `sentences` column is the subject of analysis.")
    return df
def read_sentence_from(data_uploaded, column_name="sentences") -> pd.Series:
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
    
class SentimentManager:
    def __init__(self, api_key):
        # print(f"token: {token}")
        # print(f"api_key: {api_key}")
        self.api_key = api_key
        self.aim = AIManager(self.api_key)
        self.log_manager = LogManager()
        self.messages_prompt = []

    def add_message(self, role, content):
        self.messages_prompt.append({"role": role, "content": content})

    def analyze_sentiment(self, keyword:str, sentence:str) -> float:
        print(f"keyword {keyword}, sentence {sentence}")
        self.add_message("assistant", "You are a highly skilled sentiment analyst")
        self.add_message("user", f"Analyze the sentiment of the following text: "
                                 f"'{sentence}' regarding {keyword} on a scale of 1 to 5, where 1 is very negative and 5 is very positive."
                                 f"Just tell me the score only")
        bot_response = self.aim.get_text_from_gpt(self.messages_prompt)
        # print(bot_response)
        return float(bot_response)

    def analyze_sentences(self, input_sentences, keywords):
        dict_analyzed_results = {}
        for i, sentence in enumerate(input_sentences):
            dict_analyzed_scores = dict()
            for keyword in keywords:
                st.write(keyword)
                # print(f"{i}_{keyword}")
                dict_analyzed_scores.update({keyword: self.analyze_sentiment(keyword, sentence)})
                st.write(dict_analyzed_scores)
            dict_analyzed_results[f"{i}_{sentence}"] = dict_analyzed_scores
        return dict_analyzed_results



def plot_distribution(df):
    sns.set_style("white")
    num_columns = len(df.columns) 
    num_rows =  1# 각 행에 하나의 그래프를 배치
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(10, 4))

    for i, column in enumerate(df.columns):
        sns.histplot(df[column], kde=True, label=column, bins=5, ax=axes[i])
        axes[i].set_xlabel("Feature Value")
        axes[i].set_ylabel("Density")
        axes[i].set_title(f"Distribution of {column}")
        axes[i].legend()
        sns.despine()

    plt.tight_layout()
    #return fig
    st.pyplot(fig, use_container_width=True)


# def plot_wordcloud(nouns):
#     # Create a WordCloud object with the desired settings
#     wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(nouns))
#     # Create a Matplotlib figure and axis
#     plt.figure(figsize=(8, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     # Display the Matplotlib figure within Streamlit
#     #returnfig
#     st.pyplot(plt, use_container_width=True)
