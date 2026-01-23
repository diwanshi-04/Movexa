import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{data}");
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("assets/bg.png")


df = pd.read_csv("dataset/tmdb_5000_movies.csv")
df['overview'] = df['overview'].fillna("")

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(movie):
    idx = indices[movie]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

st.title("🎬 Movie Recommendation System")

movie_list = df['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")

    for movie in recommendations:
        st.markdown(
            f"<p style='color:black; font-size:18px;'>👉 {movie}</p>",
            unsafe_allow_html=True
        )


st.markdown("""
<p style="text-align:center; font-size:22px;">
Made with <span style="color:red;">❤️</span>
</p>
""", unsafe_allow_html=True)
