import pandas as pd

from functions.aimanager import AIManager
from functions.logmanager import LogManager
import streamlit as st

def sample_sentences():
    sentences= [
                "I see the calibration settings page on each TV’s review. Are those the calibration settings that you’re saying should be used for actual viewing, or are they specifically designed for consistency during the testing process? I probably won’t pay a professional to calibrate my screen, but if there are some 'no brainer' settings to tweak, I want to be sure I’m doing it.",
                "The chart in your video has the S90C brightness numbers a lot higher compared to Rtings as well.So when looking at the comparison between S90C and A95L the chart on HDTVTest and if you would make one of the Rtings numbers both tell pretty much the same story. A95L has the upper hand in window sizes <10%, then at 10% things get more equal and at 25% and higher the peak brightness is basically identical. So the source you posted doesn’t contradict the Rtings measurements, rather it backs them up. HDTVTest for some reason just has higher numbers for all TVs, maybe a difference in how he measures things.",
                "So we did double check the brightness measurements and got very similar results. I can manage to get flashes closer to 1500 nits but they don’t stay that bright once the tv has warmed up. For what it’s worth, based on Vincent’s charts, it looks like his unit was around 1550 nits and our unit was around 1450. At high brightness levels, 100 nits won’t be super noticeable so it’s also quite possible that both units are within the expected tolerance. All in all, it remains one of the overall brightest OLED’s we’ve tested with great EOTF tracking. Hope that helps!"
    ]


    df=  pd.DataFrame({"sentences": sentences})
    st.markdown("**Supported Formats: CSV, Excel, Text**")
    st.markdown("Excel (or CSV) Considerations: `sentences` column is the subject of analysis.")
    return df


class SentimentManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.aim = AIManager(self.api_key)
        self.log_manager = LogManager()
        self.messages_prompt = []

    def add_message(self, role, content):
        self.messages_prompt.append({"role": role, "content": content})

    def reset_message(self):
        self.messages_prompt =[]
    def analyze_sentiment(self, keyword:str, sentence:str) -> float:
        try:
            # print(f"keyword {keyword}, sentence {sentence}")
            self.add_message("assistant", "You are a highly skilled sentiment analyst")
            self.add_message("user", f"Analyze the sentiment of the following text: "
                                     f"Rate the '{keyword}' in the sentence '{sentence}' on a scale from 0 (strongly negative) to 10 (strongly positive)."
                                     f"only respond as only number")
            bot_response = self.aim.get_text_from_gpt(self.messages_prompt)
            # print(f"bot_response: {bot_response}")
            bot_response = float(bot_response)
        except Exception as e:
            bot_response = 5.0

        self.reset_message()  # 리셋
        return bot_response

    def analyze_sentences(self, input_sentences:list, keywords: list):
        # print(keywords)
        # print(input_sentences)
        dict_analyzed_scores = dict()
        df_scores_list = []
        # df_scores = pd.DataFrame(columns=keywords)

        for i, sentence in enumerate(input_sentences):
            dict_scores = {keyword: self.analyze_sentiment(keyword, sentence) for keyword in keywords}
            dict_analyzed_scores[f"{i}_{sentence}"]= dict_scores
            # print(f"{i}_{sentence}: {keyword} - {score}" for keyword, score in dict_scores.items())
            # print(f"{i}_{sentence}: {dict_scores}")  # Corrected print statement

            df = pd.DataFrame.from_dict(dict_scores, orient='index')
            df_scores_list.append(df)  # Append each DataFrame to the list
        df_scores = pd.concat(df_scores_list, axis=1).T  # Concatenate all DataFrames in the list
        df_scores.reset_index(drop=True, inplace=True)  # Reset row index

        # print(df_scores)
        return df_scores




