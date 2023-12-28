from functions.gpt_assistant import GPTAssistant
import streamlit as st
def main():
    st.title("Senti-GPT")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    with st.sidebar:
        # st.subheader("Setting")
        st.session_state["apikey"] = st.text_input("GPT API KEY")
        st.session_state["role"] = st.text_input(label="Assistant ROLE",
                                                 value="You are an economist and a risk manager with excellent skills.")
        st.session_state["model"] = st.radio(label="model",
                                             options=["gpt-4-1106-preview","gpt-4-32k","gpt-3.5-turbo-1106"])
        st.session_state["temp"] = st.slider("Temperature", min_value=0.5, max_value=1.0, value=0.8, step=0.1)

        with st.expander("Profile", expanded=False):
            st.write(
                """  
                Written by TJ.Kim ☕
                """
            )
    try:
        gpt_assistant = GPTAssistant(api_key=st.session_state["apikey"],
                                     role=st.session_state["role"],
                                     temp=st.session_state["temp"])
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                full_response = gpt_assistant.generate_response(response_messages)
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    except:
        if st.session_state["apikey"] == "":
            st.error("🚨 Input your API KEY")

if __name__ == "__main__":
    main()