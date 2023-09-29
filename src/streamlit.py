import requests
import streamlit as st
import joblib
import util as utils

# Load config file
config = utils.load_config()

# Title
st.title("Bank Marketing")

# Sub-header
st.subheader("sub-header")

# Create input form
with st.form(key="data_from"):

    age = st.number_input(
        label="1.\tEnter age:",
        min_value=18,
        max_value=95,
        help=f"Value range from 18 to 95"
    )

    job = st.selectbox(
        label="2.\tEnter job:",
        options=["blue-collar", "student", "retired", "other"]
    )

    marital = st.selectbox(
        label="3.\tEnter marital status:",
        options=["single", "married", "divorced"]
    )

    education = st.selectbox(
        label="4.\tEnter education level:",
        options=["primary", "secondary", "tertiary", "unknown"]
    )

    default = int(
        st.checkbox(
            label="5.\tHas default loan?"
        )
    )
    if default:
        default = "yes"
    else:
        default = "no"

    balance = st.number_input(
        label="6.\tBalance in bank account:",
        min_value=-8019,
        max_value=102127,
        help=f"Value range from -8019 to 102127"
    )

    housing = int(
        st.checkbox(
            label="7.\tHas housing loan?"
        )
    )
    if housing:
        housing = "yes"
    else:
        housing = "no"

    loan = int(
        st.checkbox(
            label="8.\tHas other loan?"
        )
    )
    if loan:
        loan = "yes"
    else:
        loan = "no"

    contact = st.selectbox(
        label="9.\tEnter contact category:",
        options=["cellular", "telephone", "unknown"]
    )

    day = st.number_input(
        label="10.\tEnter number of days passed after last contact:",
        min_value=1,
        max_value=31,
        help=f"Value range from 1 to 31"
    )

    month = st.selectbox(
        label="11.\tEnter last contact month:",
        options=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )

    duration = st.number_input(
        label="12.\tEnter duration of call:",
        min_value=0,
        max_value=4918,
        help=f"Value range from 0 to 4918"
    )

    campaign = st.number_input(
        label="13.\tEnter number of contact during this campaign:",
        min_value=1,
        max_value=63,
        help=f"Value range from 1 to 63"
    )

    pdays = st.number_input(
        label="14.\tEnter number of days passed after last contact from previous campaign (if never, fill -1):",
        min_value=-1,
        max_value=871,
        help=f"Value range from -1 to 871"
    )

    previous = st.number_input(
        label="15.\tEnter number of contact before this campaing:",
        min_value=0,
        max_value=275,
        help=f"Value range from 0 to 275"
    )

    poutcome = st.selectbox(
        label="16.\tEnter outcome of the previous marketing campaign:",
        options=["success", "failure", "unknown", "other"]
    )

    # Create button to submit form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediciton server ..."):
            res = requests.post("http://127.0.0.1:8080/predict", json=raw_data).json()

        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error occurs while predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Client will not subscribe a term deposit.":
                st.success("Client will subscribe a term deposit.")
            else:
                st.warning("Client will not subscribe a term deposit.")