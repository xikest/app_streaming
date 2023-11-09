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

def read_df_from(data_uploaded, column_name="sentences") -> pd.Series:
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
    return df
