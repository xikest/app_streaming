import os
import pickle

class LogManager:
    def __init__(self):
        self.messages_prompt = []
        self.log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def add_message(self, role, content):
        self.messages_prompt.append({"role": role, "content": content})



    def save_log(self, log_file):
        log_file = log_file+".pkl"
        log_path = os.path.join(self.log_dir, log_file)
        with open(log_path, "wb") as file:
            pickle.dump(self.messages_prompt, file)


    def load_log(self, log_file):
        log_path = os.path.join(self.log_dir, log_file)
        try:
            with open(log_path, "rb") as file:
                self.messages_prompt = pickle.load(file)
        except (FileNotFoundError, EOFError):
            self.messages_prompt = []
            self.add_message("assistant", "you understand all inputs in English. Respond with a 20-word answer in Korean.")
            self.save_log(log_file)

            # self.add_message("assistant", "You are a thoughtful assistant, and you understand all inputs in English. Respond to all input in 20 words and answer in korea")
            # self.save_log(log_file)

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
