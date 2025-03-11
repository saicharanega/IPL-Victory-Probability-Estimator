git add .# Importing the libraries
import streamlit as st
import pickle as pkl
import pandas as pd

# Setting up a wide page layout
st.set_page_config(layout="wide")

# Title of the page
st.title("IPL Win Predictor")

# Importing data and model from pickle files
teams = pkl.load(open('team.pkl', 'rb'))
cities = pkl.load(open('city.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

# First Row and columns
col1, col2, col3 = st.columns(3)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))
with col3:
    selected_city = st.selectbox('Select the host city', sorted(cities))

target = st.number_input('Target Score', min_value=0, max_value=720, step=1)

col4, col5, col6 = st.columns(3)
with col4:
    score = st.number_input('Score', min_value=0, max_value=720, step=1)
with col5:
    overs = st.number_input('Overs Done', min_value=0, max_value=20, step=1)
with col6:
    wickets = st.number_input('Wickets Fell', min_value=0, max_value=10, step=1)

# Handling edge cases before prediction
if overs == 0:
    st.error("Overs done cannot be zero!")
elif st.button('Predict Probabilities'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    remaining_wickets = 10 - wickets

    # Avoid division by zero for Current Run Rate (CRR) and Required Run Rate (RRR)
    if overs > 0:
        crr = score / overs
    else:
        crr = 0

    if balls_left > 0:
        rrr = (runs_left * 6) / balls_left
    else:
        rrr = 0

    # Create a DataFrame to pass to the model
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'Score': [score],
        'Wickets': [remaining_wickets],
        'Remaining Balls': [balls_left],
        'target_left': [runs_left],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Ensure the input DataFrame matches the expected format
    st.write("Input DataFrame to model:", input_df)

    # Check if column names match the model's expected feature names
    if hasattr(model, 'feature_names_in_'):
        st.write("Model feature names:", model.feature_names_in_)

    # Predict probabilities
    try:
    # Predict probabilities
        result = model.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.header(f"{batting_team} - {round(win * 100)}%")
        st.header(f"{bowling_team} - {round(loss * 100)}%")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write(f"Error Details: {e}")
