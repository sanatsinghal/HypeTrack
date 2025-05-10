import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="App Similarity Checker", layout="centered")
st.title("📲 HypeTrack")

# --- Track submission status ---
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# --- App Submission Form ---
if not st.session_state.submitted:
    st.markdown("Compare your app to other leading apps using App Store data and social engagement metrics to evaluate its success potential")

    app_name = st.text_input("App Name")
    category = st.selectbox("Category", ["Music", "Video Streaming", "Podcasts", "Other"])
    developer = st.text_input("Developer / Team Name")
    price = st.selectbox("Planned Price", ["Free", "Paid"])
    update_frequency_days = st.number_input("Expected Update Frequency (in Days)", min_value=1)
    lines_of_code = st.number_input("Estimated Lines of Code", min_value=1000)
    description = st.text_area("Brief Description of the App")

    if st.button("Submit App"):
        st.session_state.submitted = True
        st.session_state.app_data = {
            "app_name": app_name,
            "category": category,
            "developer": developer,
            "price": price,
            "update_frequency_days": update_frequency_days,
            "lines_of_code": lines_of_code,
            "description": description
        }

# --- Show Results After Submission ---
if st.session_state.submitted:
    app_data = st.session_state.app_data

    new_app = pd.DataFrame([app_data])
    st.success("✅ App submitted successfully!")

    # 📋 App Summary
    st.markdown("### 📋 App Submission Summary")
    st.markdown(f"""
    - 🆔 **App Name:** `{app_data["app_name"]}`  
    - 🎯 **Category:** `{app_data["category"]}`  
    - 👥 **Developer:** `{app_data["developer"]}`  
    - 💲 **Planned Price:** `{app_data["price"]}`  
    - 🔁 **Update Frequency:** `{app_data["update_frequency_days"]}` days  
    - 🧮 **Lines of Code:** `{app_data["lines_of_code"]:,}`  
    - 📝 **Description:**  
    > {app_data["description"]}
    """)

    # --- Compare with Existing Data ---
    if os.path.exists("appstore.csv") and os.path.exists("twitter.csv"):
        appstore_df = pd.read_csv("appstore.csv")
        twitter_df = pd.read_csv("twitter.csv")
        merged_df = pd.merge(appstore_df, twitter_df, on="app_name", how="inner")

        compare_df = merged_df[[
            "app_name", "category", "developer", "price",
            "update_frequency_days", "lines_of_code"
        ]].copy()

        compare_df = pd.concat([compare_df, new_app.drop(columns="description")], ignore_index=True)

        le_category = LabelEncoder()
        le_price = LabelEncoder()
        compare_df["category_enc"] = le_category.fit_transform(compare_df["category"])
        compare_df["price_enc"] = le_price.fit_transform(compare_df["price"])

        features = compare_df[["category_enc", "price_enc", "update_frequency_days", "lines_of_code"]]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # KNN Model
        n_neighbors = min(6, len(compare_df))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(scaled_features)
        distances, indices = knn.kneighbors([scaled_features[-1]])

        similar_indices = indices[0][1:n_neighbors]
        similar_apps = merged_df.iloc[similar_indices].copy()
        similar_apps["similarity_score"] = 1 / (1 + distances[0][1:n_neighbors])
        similar_apps = similar_apps.sort_values(by="similarity_score", ascending=True)

        # Dropdown: Social Metric
        metric_choice = st.selectbox(
            "Compare Similarity Against:",
            ["Twitter Mentions", "Twitter Likes", "Sentiment Score"]
        )

        metric_map = {
            "Twitter Mentions": ("twitter_mentions_weekly", "Mentions", "mediumorchid"),
            "Twitter Likes": ("twitter_likes", "Likes", "lightcoral"),
            "Sentiment Score": ("sentiment_score", "Sentiment Score", "limegreen")
        }

        metric_col, metric_label, metric_color = metric_map[metric_choice]

        # --- Graph 1: Similarity Score ---
        st.markdown("### 🟦 Similarity Score (Top Similar Apps)")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.barh(similar_apps["app_name"], similar_apps["similarity_score"], color="skyblue")
        ax1.set_xlabel("Similarity Score (higher = more similar)")
        ax1.set_ylabel("App Name")
        ax1.set_title("Similarity Score")
        ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
        st.pyplot(fig1)

        # --- Graph 2: Selected Twitter Metric ---
        st.markdown(f"### 🟩 {metric_label} (Top Similar Apps)")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        bars = ax2.barh(similar_apps["app_name"], similar_apps[metric_col], color=metric_color)
        ax2.set_xlabel(metric_label)
        ax2.set_ylabel("App Name")
        ax2.set_title(metric_label)
        ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

        if metric_choice == "Sentiment Score":
            ax2.set_xlim(-1, 1)

        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', fontsize=8)

        st.pyplot(fig2)

        # --- Social Averages ---
        avg_similarity = similar_apps["similarity_score"].mean()
        avg_mentions = similar_apps["twitter_mentions_weekly"].mean()
        avg_likes = similar_apps["twitter_likes"].mean()
        avg_sentiment = similar_apps["sentiment_score"].mean()

        st.markdown("### 📊 Estimated Social Potential")
        st.write(f"📈 **Avg Weekly Mentions:** {int(avg_mentions)}")
        st.write(f"💬 **Avg Sentiment Score:** {avg_sentiment:.2f}")
        st.write(f"❤️ **Avg Twitter Likes:** {int(avg_likes)}")

        normalized_mentions = min(avg_mentions / 10000, 1.0)
        normalized_likes = min(avg_likes / 20000, 1.0)
        normalized_sentiment = (avg_sentiment + 1) / 2

        success_score = (
            (avg_similarity * 0.4) +
            (normalized_mentions * 0.25) +
            (normalized_likes * 0.2) +
            (normalized_sentiment * 0.15)
        )

        st.markdown("### 🧠 AI-Powered Success Prediction")
        st.write(f"📊 **Predicted Success Score:** `{success_score:.2f}` _(scale: 0 to 1)_")

        if success_score >= 0.75:
            st.success("🚀 Strong likelihood of success! This app resembles high-performing, well-received apps.")
        elif success_score >= 0.5:
            st.info("🧐 Moderate potential. Consider refining marketing, features, or niche focus.")
        else:
            st.warning("⚠️ Low predicted performance. You may want to rethink features, audience, or strategy.")
    else:
        st.warning("⚠️ Missing data: Make sure both `appstore.csv` and `twitter.csv` are in your folder.")

    # --- Resubmit Button ---
    st.markdown("---")
    if st.button("🔁 Submit Another App"):
        st.session_state.submitted = False
        st.rerun()
