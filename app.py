
import pickle
import streamlit as st

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("SMS Spam Detection")
st.write("Check whether your message is Spam or Ham.")

message = st.text_area("Enter your SMS message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_tfidf = tfidf.transform([message])
        prediction = model.predict(message_tfidf)

        if prediction[0] == "spam":
            st.error("This message is SPAM!")
        else:
            st.success("This message is HAM (Not Spam).")
