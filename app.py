import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import math

# Load models
@st.cache_resource
def load_models():
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    return knn_model, kmeans_model

knn_model, kmeans_model = load_models()

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('anime.csv')  # Update with your dataset path
    return df

df = load_data()

# Load CSS
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("styles.css")

# Sidebar options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Team", "About"])

# Streamlit UI
if page == "Home":
    st.title('Anime Recommender System')
    st.write("Welcome to the Anime Recommender System! Enter your ratings below to get recommendations.")

    # User input
    user_id = st.text_input("Enter your user ID:")
    anime_titles = df['name'].unique()
    selected_animes = st.multiselect("Select anime titles you have watched and rate them", options=anime_titles)

    ratings = {}
    for anime in selected_animes:
        rating = st.slider(f"Rate {anime}", 1, 10, 5)
        ratings[anime] = rating

    # Predict recommendations
    if st.button('Get Recommendations'):
        if user_id and ratings:
            # Create a DataFrame for user ratings
            user_ratings = pd.DataFrame(list(ratings.items()), columns=['name', 'user_rating'])
            
            # Merge user ratings with the original dataset
            user_ratings = pd.merge(user_ratings, df[['anime_id', 'name']], on='name', how='left')

            # Add the user_id column
            user_ratings['user_id'] = user_id

            # Debugging: Check the columns of user_ratings after merge
            st.write("Merged User Ratings DataFrame:")
            st.write(user_ratings)

            # Example processing for KNN (if KNN is based on collaborative filtering)
            knn_predictions = knn_model.predict(user_ratings[['user_id', 'anime_id']])

            # Example processing for KMeans (cluster assignment based predictions)
            kmeans_predictions = kmeans_model.predict(user_ratings[['user_id', 'anime_id']])

            # Combine predictions (using an average strategy)
            combined_predictions = (knn_predictions + kmeans_predictions) / 2

            # Prepare and display recommendations
            user_ratings['predicted_rating'] = combined_predictions
            recommended_animes = user_ratings[['name', 'predicted_rating']].sort_values(by='predicted_rating', ascending=False)

            st.write("Recommended Anime for you:")
            st.write(recommended_animes)

        else:
            st.write("Please enter your user ID and select anime titles.")

elif page == "Team":
    st.header('The Team')
    st.write("""
        Welcome to the simple anime recommender system!

        Our mission is to help you find animes that you'll love based on your preferences and ratings.

        This system uses collaborative filtering to suggest animes you haven't seen yet but might enjoy based on the ratings of other users with similar tastes.

        **Team Members:**
        - Neo Modibedi: Team Leader
        - Melody Msimango: Project Manager
        - Priscilla Matuludi: GitHub Manager
        - Keryn Rabe: Organizer
        - Tumelo Matamane: Team Member
        - Ntandoyenkosi Biyela: Team Member

        Enjoy your anime recommendations!
    """)

elif page == "About":
    st.header("About this Project")
    st.write("""
        This Anime Recommender System was built to help users find new animes based on their preferences and past ratings.

        The system leverages collaborative filtering techniques and various machine learning models to generate recommendations.

        We hope you enjoy using this system and find it helpful in discovering new animes to watch!
    """)

# Footer
st.write("Thank you for using our Anime Recommender System!")
