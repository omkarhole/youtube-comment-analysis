import os ,re
import streamlit as st
from urllib.parse import urlparse, parse_qs
import joblib 
import pandas as pd
import numpy as np
import plotly.express as px
# from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


st.set_page_config(page_title="YouTube Comment Analysis", page_icon=":bar_chart:", layout="wide")

st.title("📊 YouTube Comment Sentiment Analysis")
st.caption(
    "Analyze YouTube comments using your trained "
    "TF-IDF and Logistic Regression pipeline."
)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load('sentiment_pipeline.pkl')

try:
    model=load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()





# =========================================================
# 3. YOUTUBE API KEY
# =========================================================

def get_saved_api_key():
    try:
        api_key=st.secrets.get("YOUTUBE_API_KEY","")
        if api_key:
            return api_key
    except Exception as e:
        st.error(f"Error retrieving API key: {e}")
        return None

    return os.getenv("YOUTUBE_API_KEY", "")

saved_api_key = get_saved_api_key()

with st.sidebar:

    api_key=saved_api_key 
    st.divider()
    st.header("Select Range ")


    max_comments = st.select_slider(
        "Comments to analyze",
        options=[50, 100, 200, 300, 500,1000],
        value=200
    )

    comment_order = st.selectbox(
        "Comment order",
        options=["time", "relevance"],
        format_func=lambda x: {
            "time": "Newest first",
            "relevance": "Most relevant"
        }[x]
    )

    st.divider()

    st.caption("Model: TF-IDF + Logistic Regression")
    st.caption("Task: Positive / Negative / Neutral")
    st.caption("Created by: Omkar Hole")
    st.markdown("[GitHub](https://github.com/omkarhole)")

# =========================================================
# 4. EXTRACT YOUTUBE VIDEO ID
# =========================================================

def extract_video_id(url):
    url = url.strip()

    # Direct YouTube video ID
    if re.fullmatch(r"[\w-]{11}", url):
        return url

    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower().split(":")[0]
        host = host.removeprefix("www.")

        if host in ("youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [None])[0]

            if parsed.path.startswith(
                ("/embed/", "/shorts/", "/live/")
            ):
                parts = parsed.path.strip("/").split("/")
                if len(parts) >= 2:
                    return parts[1]

        if host == "youtu.be":
            return parsed.path.strip("/").split("/")[0]

    except Exception:
        return None

    return None

# =========================================================
# 5. CONNECT TO YOUTUBE
# =========================================================

def create_youtube_client(key):
    return build(
        "youtube",
        "v3",
        developerKey=key,
        cache_discovery=False
    )


# =========================================================
# 6. FETCH VIDEO DETAILS
# =========================================================

def fetch_video_details(youtube, video_id):
    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    if not response.get("items"):
        raise ValueError(
            "Video not found. Check the video URL or ID."
        )

    video = response["items"][0]
    snippet = video.get("snippet", {})
    stats = video.get("statistics", {})

    return {
        "title": snippet.get("title", "Unknown"),
        "channel": snippet.get("channelTitle", "Unknown"),
        "published_at": snippet.get("publishedAt", ""),
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "total_comments": int(stats.get("commentCount", 0))
    }


# =========================================================
# 7. FETCH YOUTUBE COMMENTS
# =========================================================

def fetch_youtube_comments(youtube, video_id, limit, order):
    records = []
    next_page_token = None

    while len(records) < limit:
        batch_size = min(100, limit - len(records))

        request_args = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": batch_size,
            "order": order,
            "textFormat": "plainText"
        }

        if next_page_token:
            request_args["pageToken"] = next_page_token

        response = youtube.commentThreads().list(
            **request_args
        ).execute()

        items = response.get("items", [])

        if not items:
            break

        for item in items:
            comment_data = (
                item["snippet"]["topLevelComment"]["snippet"]
            )

            records.append({
                "Comment": comment_data.get("textDisplay", ""),
                "Author": comment_data.get(
                    "authorDisplayName", "Unknown"
                ),
                "Likes": int(comment_data.get("likeCount", 0)),
                "Published At": comment_data.get("publishedAt", ""),
                "Comment ID": item["snippet"][
                    "topLevelComment"
                ].get("id", "")
            })

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return pd.DataFrame(records)


def standardize_sentiment(label):
    value = str(label).strip().lower()

    if "positive" in value or value in ("pos", "1"):
        return "Positive"

    if "negative" in value or value in ("neg", "-1"):
        return "Negative"

    if "neutral" in value or value in ("neu", "0"):
        return "Neutral"

    return str(label).strip().title()


# input form 
with st.form('youtube_analysis_form'):
    st.subheader("🎬 Analyze a YouTube Video")

    video_url = st.text_input(
        "YouTube video URL or video ID",
        placeholder="https://www.youtube.com/watch?v=XXXXXXXXXXX"
    )

    submitted = st.form_submit_button(
        "🔍 Analyze Video",
        use_container_width=True,
        type="primary"
    )

# =========================================================
# 10. ANALYZE VIDEO
# =========================================================


if submitted:
    if  not api_key:
        st.error("Please enter a YouTube API key.")
        st.stop()
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Please enter valid YouTube video URL or ID.")
        st.stop()

    try:
        with st.spinner(
            "Fetching video information and analyzing comments..."
        ):
            youtube = create_youtube_client(api_key)

            video_details = fetch_video_details(
                youtube, video_id
            )

            df = fetch_youtube_comments(
                youtube,
                video_id,
                int(max_comments),
                comment_order
            )

            if df.empty:
                st.warning(
                    "No public top-level comments were found. "
                    "Comments might be disabled or unavailable."
                )
                st.stop()
            
            # Run the trained ML pipeline
            df['Comment']=(df['Comment'].fillna("").astype(str))

            df['Sentiment'] = model.predict(df['Comment'].to_list())

            df["Sentiment"] = df["Sentiment"].apply(
                standardize_sentiment
            )

            df["Published At"] = pd.to_datetime(
                df["Published At"],
                errors="coerce",
                utc=True
            )
            st.session_state["analysis_df"] = df
            st.session_state["video_details"] = video_details
            st.session_state["video_id"] = video_id

            st.success(
            f"Analysis completed! "
            )
    except HttpError as e:
        st.error(
            "YouTube API request failed. Check your API key, "
            "API quota, video access, and comment availability."
        )
        st.code(str(e))
        st.stop()

    except Exception as e:
        st.error("Something went wrong during analysis.")
        st.code(str(e))
        st.stop()


# =========================================================
# 11. DISPLAY DASHBOARD
# =========================================================


if 'analysis_df' in st.session_state :
    df=st.session_state['analysis_df'].copy()
    details = st.session_state["video_details"]
    video_id = st.session_state["video_id"]


    # -----------------------------------------------------
    # VIDEO DETAILS
    # -----------------------------------------------------

    st.divider()
    st.header("🎬 Video Details")

    st.markdown(f"**Title:** {details['title']}")
    st.markdown(f"**Channel:** {details['channel']}")

    if details['published_at']:
        st.caption(f'Published:{details["published_at"]}')

    st.markdown(
        f"[▶ Watch on YouTube]"
        f"(https://www.youtube.com/watch?v={video_id})"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("👁️ Views", f"{details['views']:,}")
    c2.metric("👍 Likes", f"{details['likes']:,}")
    c3.metric(
        "💬 Total Video Comments",
        f"{details['total_comments']:,}"
    )

    st.divider()
    st.header("📊 Sentiment Analysis ")

    sentiment_order=['Positive','Neutral','Negative']

    counts=(
        df['Sentiment']
        .value_counts()
        .reindex(sentiment_order, fill_value=0)
    )
    total=len(df)

    positive_pct = (
        counts["Positive"] / total * 100 if total else 0
    )

    negative_pct = (
        counts["Negative"] / total * 100 if total else 0
    )

    neutral_pct = (
        counts["Neutral"] / total * 100 if total else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Analyzed", f"{total:,}")
    c2.metric(
        "😊 Positive",
        f"{counts['Positive']:,}",
        f"{positive_pct:.1f}%"
    )
    c3.metric(
        "😠 Negative",
        f"{counts['Negative']:,}",
        f"{negative_pct:.1f}%"
    )
    c4.metric(
        "😐 Neutral",
        f"{counts['Neutral']:,}",
        f"{neutral_pct:.1f}%"
    )


    # -----------------------------------------------------
    # PIE AND BAR CHARTS
    # -----------------------------------------------------

    st.subheader("📊 Sentiment Distribution")

    sentiment_df=counts.rename_axis("Sentiment").reset_index(name='Count')

    col1,col2=st.columns(2)

    with col1:
        fig_pie = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Count",
            hole=0.38,
            title="Sentiment Percentage",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#22C55E",
                "Negative": "#F97316",
                "Neutral": "#3B82F6"
            }
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="label+percent"
        )

        fig_pie.update_layout(
            margin=dict(t=50, b=10, l=10, r=10)
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with col2:
        fig_bar = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            text="Count",
            title="Sentiment Counts",
            color_discrete_map={
                "Positive": "#22C55E",
                "Negative": "#F97316",
                "Neutral": "#3B82F6"
            }
        )

        fig_bar.update_layout(
            showlegend=False,
            margin=dict(t=50, b=10, l=10, r=10),
            xaxis_title="Sentiment",
            yaxis_title="Number of Comments"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )


    # -----------------------------------------------------
    # SENTIMENT OVER TIME
    # -----------------------------------------------------

    st.divider()
    st.subheader("📌 Sentiment Over Time")

    dated_df = df.dropna(
        subset=["Published At"]
    ).copy()

    if not dated_df.empty:
        dated_df['Date']=(
            dated_df['Published At'].dt.floor("D")
        )

        daily = (
            dated_df.groupby(["Date", "Sentiment"])
            .size()
            .reset_index(name="Count")
        )
        fig_time = px.line(
            daily,
            x="Date",
            y="Count",
            color="Sentiment",
            markers=True,
            title="Daily Comment Counts by Sentiment",
            color_discrete_map={
                "Positive": "#22C55E",
                "Negative": "#F97316",
                "Neutral": "#3B82F6"
            }
        )

        fig_time.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Comments",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_time,
            use_container_width=True
        )

    else:
        st.info("No valid comment dates are available.")

    # -----------------------------------------------------
    # WORD CLOUD
    # -----------------------------------------------------

    st.divider()
    st.subheader("☁️ Word Cloud")

    all_text=" ".join(df['Comment'].dropna().astype(str)).strip()
    if all_text:

        extra_stopwords = {
            "video", "youtube", "watch", "subscribe",
            "channel", "https", "http", "www"
        }

        stopwords = set(STOPWORDS).union(extra_stopwords)

        wordcloud = WordCloud(
            width=1400,
            height=600,
            background_color="white",
            stopwords=stopwords,
            max_words=200,
            collocations=False,
            min_font_size=8
        ).generate(all_text)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:
        st.info("No comment text is available.")

    
    # -----------------------------------------------------
    # COMMENT HIGHLIGHTS
    # -----------------------------------------------------
    st.subheader("💬 Comment Highlights")

    positive_df = df[
        df["Sentiment"] == "Positive"
    ].copy()

    negative_df = df[
        df["Sentiment"] == "Negative"
    ].copy()

    if df["Comment"].notna().any():
        positive_df = positive_df.sort_values(
            "Comment", ascending=False
        )
        negative_df = negative_df.sort_values(
            "Comment", ascending=False
        )

    with st.expander("🌟 Top 5 Positive Comments"):

        if positive_df.empty:
            st.info("No positive comments predicted.")
        else:
            for i, row in enumerate(
                positive_df.head(5).itertuples(),
                start=1,
            ):
                st.markdown(
                    f"**{i}.** {row.Comment}"
                )
                st.caption(f"👍 {row.Likes} likes")
                st.divider()

    with st.expander("🚨 Top 5 Negative Comments"):

        if negative_df.empty:
            st.info("No negative comments predicted.")
        else:
            for i, row in enumerate(
                negative_df.head(5).itertuples(),
                start=1,
            ):
                st.markdown(
                    f"**{i}.** {row.Comment}"
                )
                st.caption(f"👍 {row.Likes} likes")
                st.divider()

    # -----------------------------------------------------
    # SEARCH AND FILTER COMMENTS
    # -----------------------------------------------------


    # Complete comment table
    st.subheader("📄 Analyzed Comments")

    display_df = df[
        [
            "Comment",
            "Sentiment",
            "Published At",
            "Likes",
        ]
    ].copy()

    display_df.rename(
        columns={
            "Comment": "CommentText",
            "Published At": "PublishedAt",
        },
        inplace=True,
    )

    display_df["PublishedAt"] = (
        display_df["PublishedAt"].astype(str)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # Download results
    st.download_button(
        "⬇️ Download Analyzed Comments (CSV)",
        data=df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="youtube_sentiment_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.info(
        "Paste a YouTube video URL above and click "
        "**Analyze Video** to get started."
    )