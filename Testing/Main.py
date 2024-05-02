import streamlit as st
from web3 import Web3
import uuid
from dotenv import load_dotenv

# Streamlit UI
st.title("Macro Economic ")
# Explain the purpose of the app
st.subheader("""
    This is a tester, for the purpose of testing multiple forecasting models, and comparing their results.
             """)
# Explain the purpose of the app           
st.warning("This application serves to demonstrate the capabilities of blockchain ledger technology, and is not meant to be used in a production environment.")

# Developer information
developers = [
    {
        "name": "Najib Abou Nasr",
        "socials": {
            "LinkedIn": "https://www.linkedin.com/in/najib-abou-nasr-a43520258/",
            "GitHub": "https://github.com/najibabounasr",
            "Twitter": "https://twitter.com/najib_abounasr"
        },
        "info": "Former Computer Science Major at Santa Monica College."
    },
]
    
# Display developer information
st.sidebar.title("Developers")
for developer in developers:
    st.sidebar.subheader(developer["name"])
    st.sidebar.markdown(developer["info"])
    for social, link in developer["socials"].items():
        st.sidebar.markdown(f"[{social}]({link})")

    st.sidebar.write("---")