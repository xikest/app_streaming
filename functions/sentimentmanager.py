import pandas as pd

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
                # print(f"{i}_{keyword}")
                dict_analyzed_scores.update({keyword: self.analyze_sentiment(keyword, sentence)})
            dict_analyzed_results[f"{i}_{sentence}"] = dict_analyzed_scores
        return dict_analyzed_results




