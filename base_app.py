import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from wordcloud import WordCloud  # Fixed import for word cloud visualization

# Load CSS
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("styles.css")  # Load CSS before any other code

# Load models
@st.cache_resource
def load_models():
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    return svd_model, knn_model, kmeans_model

svd_model, knn_model, kmeans_model = load_models()

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('anime.csv')  # Update with your dataset path
    return df

df = load_data()

# Sidebar options
st.sidebar.title("navigation")
page = st.sidebar.radio("explore", ["home", "recommender", "eda", "team"])

# Streamlit UI
if page == "home":
    st.title('Welcome to our "AnimeXplore" App!')
    st.markdown("""
    ## Overview
    Welcome to our recommender systems where you can get personalized recommendations, explore data insights, and learn more about our team.

    ### Navigation
    - **Recommender page**: Get personalized recommendations based on your preferences.
    - **EDA page**: Dive into the data and explore various visualizations.
    - **About team page**: Learn more about the team behind this app.

    ### How to use
    1. Navigate to the recommender page to start getting recommendations.
    2. Explore the EDA page to understand the data behind the recommendations.
    3. Visit the about team page to meet our team.

    We hope you enjoy using this system and find it helpful in discovering new animes to watch!
    """)

elif page == "recommender":
    st.title('Anime Recommender System')
    st.write("Welcome to the anime recommender system! Enter your ratings below to get recommendations.")

    # User input
    user_id = st.text_input("Enter your user id:")
    anime_titles = df['name'].unique()
    selected_animes = st.multiselect("Select anime titles you have watched and rate them", options=anime_titles)

    ratings = {}
    for anime in selected_animes:
        rating = st.slider(f"Rate {anime}", 1, 10, 5)
        ratings[anime] = rating

    # Predict recommendations
    if st.button('Get recommendations'):
        if user_id and ratings:
            user_ratings = pd.DataFrame(list(ratings.items()), columns=['name', 'user_rating'])

            # Merge user ratings with the original dataset
            user_ratings = pd.merge(user_ratings, df, on='name', how='left')

            # Debugging: check the columns of user_ratings after merge
            st.write("Merged user ratings dataframe:")
            st.write(user_ratings)

            try:
                # Ensure 'user_rating' exists before proceeding
                if 'user_rating' in user_ratings.columns:
                    user_ratings_matrix = user_ratings[['user_rating']].values.reshape(-1, 1)
                    svd_predictions = svd_model.transform(user_ratings_matrix).flatten()
                    knn_predictions = knn_model.predict(user_ratings_matrix)
                    kmeans_predictions = kmeans_model.predict(user_ratings_matrix)

                    # Combine predictions
                    combined_predictions = (svd_predictions + knn_predictions + kmeans_predictions) / 3

                    # Prepare and display recommendations
                    user_ratings['predicted_rating'] = combined_predictions
                    recommended_animes = user_ratings[['name', 'predicted_rating']].sort_values(by='predicted_rating', ascending=False)

                    st.write("Recommended anime for you:")
                    st.write(recommended_animes)
                else:
                    st.error("The 'user_rating' column was not found in the merged DataFrame.")

            except Exception as e:
                st.error(f"Error generating predictions: {e}")

        else:
            st.write("Please enter your user ID and select anime titles.")

elif page == "eda":
    st.title("Exploratory Data Analysis")
    st.subheader("Genre Word Cloud")

    if 'genre' in df.columns:
        genres_series = df['genre'].dropna()  # Ensure there are no nans
        all_genres = ' '.join(genres_series)

        # Generate the word cloud
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(all_genres)

        # Plotting the word cloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.write("The dataset does not contain a 'genre' column.")

elif page == "team":
    st.header('The Team')
    st.write("""
        Welcome to the simple anime recommender system!

        Our mission is to help you find animes that you'll love based on your preferences and ratings.

        This system uses collaborative filtering to suggest animes you haven't seen yet but might enjoy based on the ratings of other users with similar tastes.

        **Team members:**
        - Neo Modibedi: Team Leader
        - Melody Msimango: Project Manager
        - Priscilla Matuludi: GitHub Manager
        - Keryn Rabe: Organizer
        - Tumelo Matamane: Team Member
        - Ntandoyenkosi Biyela: Team Member

        Enjoy your anime recommendations!
    """)

# Footer
st.write("Thank you for using our anime recommender system!")
