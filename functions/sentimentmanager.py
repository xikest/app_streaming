import pandas as pd
from wordcloud import WordCloud
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from functions.aimanager import AIManager
from functions.logmanager import LogManager


def sample_sentences() -> pd.DataFrame:
    # creating exmaple data
    sentences = {
        'sentences': [
             "My 12 year old Vizio tv died two weeks ago and we went out and spent about $700 on a nice name brand 55 inch tv. 12 years makes a huge difference in tv picture quality! We were blown away with that TV. Then, the other day, this pops up.The picture is truly stunning out of the box and might be a bit too vivid for everyday watching, but hey, there are a ton of settings that you can adjust to your preference. The border is small so the tv just looks like a movie theater. Let's just say this is the best picture I've ever seen, and I was tv shopping just two weeks ago. Specs are 4K: 3,840 x 2,160 pixels. Upscaling is very good. Not every source or movie is full screen and I changed the setting to that.Settings: Take your time and go though the settings. Level 1 - image and sound. Try out the different setting to find your favorites. Level 2 - privacy and under the hood. Without using Google, this is easier but check out BRAVIA, power savings, bluetooth, accessibility and parental controls. I didn't hook up the included camera but Gesture Control seems intriguing. I haven't look at what else is connected to the camera, but personally I wasn't interested. Please check it out for yourself.
The absolute best feature is Power on Behavior instead of having to go through the Sony menu everytime, I've set it to HDMI 1 ( AppleTV ) and that's all I see. Brilliant.. The remote - blah. In this day and age who uses batteries? The other brand had a solar panel on the back and my apple tv remote is rechargeable. Fail. It's also pretty big but the layout and buttons are clear
Smart Tv - this is my big objection to smart things - privacy. According to the PDF  User must accept Google Terms of Service (http://www.google.com/policies/terms/), Play Terms of Service (https://play.google.com/intl/en-US_us/about/play-terms/index.html) and Privacy Policy (http://www.google.com/policies/privacy/) to use TV. User must connect to a Google account to use certain advertised features, including voice to activate linked apps, and install certain apps and operating software during setup. Use of TV without connecting to a Google account allows only basic TV features and certain apps  I don't need Google tracking everything I watch or do with my tv. If you love Google, go ahead and sign up or use your Google account.
This is compatible with with Airplay and Homekit by Apple Now, the good thing is that you can turn off most of it,including the microphone via a switch in the back. I use an Apple Tv ( which probably tracks everything I do ) so I'm using a single source for apps. If you don't have an Appletv kind of device, having access to Apps would be nice.
I'm not a gamer but the specs look just fine for that. Ports: Plenty of them plus not one but two USB ports, ethernet and bluetooth support. Etc: I can't seem to get my Apple Homepod speakers to pair up, nor get a bluetooth device AND the tv speakers to work at the same time. I like the stand, it has standard 300 x 300 VESA bolt pattern for hanging with a wall mount kit.
Future updates: Sony promises updates via firmware like  Hands Free feature ,  Dolby Vision Gaming ,  Screen Size feature  and  Multi View feature . We'll see. Overall, the picture is what you're after, and this has it nailed.",
             "Wow. Seriously.
I don't know what voodoo Sony used to pull this off, but the picture on this TV right out of the box is INSANE. It has depth to it like I've never seen out of a television; almost like it's 3D. It looks like you could reach right into the screen and grab something - difficult to explain without seeing it in person.
I was not impressed with the Amazon delivery service though - they didn't even wear gloves when setting it up and now I'm dealing with smudges, ugh.
But the TV itself is unreal I've never seen something this good before. Totally glad I spent the extra $$ to get this. A+
Edit: Forgot to mention that this TV runs 4K120 & VRR flawlessly",
             "Of course I was excited to order a beautiful SONY OLED TV, but I had no idea it would be this fantastic. My current TV was an off brand bought at a warehouse store less than a year ago. It was having intermittent issues and I was passively looking for the latest and greatest. I decided if I wanted to actually go for a NEW tv, I wanted to go ALL IN and get something with all of the bells and whistles. This TV certainly offers that.
Firstly, lets talk about the picture. It is INCREDIBLE. I have the 55 inch and everything looks just perfect. I am absolutely amazed at the brightness, and clarity and the inky blacks. Everything is on point. But add to that the incredible sound! I am not using a soundbar, but you definitely wouldn't know it! I went into the Bravia menu and chose the IMAX demo... and the BASS!!! OH MY! it is fantastic! And there are so many settings to tweak it this way or that, enhance the voices etc. So lets just say the sound and picture are everything you would want in a HIGH END TV and MORE!
But now lets talk about that little camera that comes with. Dont worry, you don't HAVE to plug it in. But I did. I decided, again, to go all in and select YES to all of the features and options, (privacy concerns be darned! I wanted FEATURES!) Well, I got features! I can hold my palm up and close my fist and my TV turns off!!! I can hold my palm up and move it up to raise the volume. There are gestures to pause.. fast forward.. etc. It is stupid awesome. Stupid because, well, do I need it? no. But awesome because it is just that! AWESOME! I can pause in a hurry (and quietly) if the remote is across the room. Know what else the camera does? It knows how far you are and adjusts the TV brightness and color accordingly! If you are watching from the side, it adjusts the sound balance, phasing and acoustics accordingly. (they call it Ambient Optimization Pro) and it dims your tv saving power when you leave the room. Sony says any information collected with Bravia cam is never shared with Sony or third parties. It also comes with a shutter on the camera that you can close. Ohh you can also do zoom calls on your TV! SO when I said I wanted a feature rich TV... boy did I ever get that! I ABSOLUTELY HIGHLY RECOMMEND this beautiful TV for all of your Techy/Nerdy/Gadgety/Movie Viewing Needs! Sooooo satisfying!" ]

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


def plot_wordcloud(nouns):
    # Create a WordCloud object with the desired settings
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(nouns))
    # Create a Matplotlib figure and axis
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # Display the Matplotlib figure within Streamlit
    #returnfig
    st.pyplot(plt, use_container_width=True)


