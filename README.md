# 📊 YouTube Comment Sentiment Analysis

### Turn YouTube comments into useful audience insights

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-TF--IDF%20%2B%20Logistic%20Regression-6C5CE7)](#-machine-learning-pipeline)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)](#)

**Analyze public YouTube comments, predict sentiment, and explore the results in an interactive dashboard.**

[Features](#-features) · [Demo](#-live-demo) · [Setup](#-run-locally) · [Deployment](#-deploy-on-streamlit-community-cloud) · [How It Works](#-how-it-works) · [Troubleshooting](#-troubleshooting)

</div>

---

## 📌 About the Project

YouTube videos can receive hundreds or thousands of comments. Reading them one by one takes time. This project uses a trained machine-learning pipeline to classify fetched comments as **Positive**, **Negative**, or **Neutral**, then presents the results in a Streamlit dashboard.

The app uses an existing `sentiment_pipeline.pkl` model. It does **not** train a new model when the dashboard runs.

> **Note:** Predictions depend on the quality and training data of the saved model. Treat the results as estimates, not perfect labels.

## ✨ Features

| Feature | Description |
|---|---|
| 🎬 Video details | Shows the video title, channel, views, likes, and reported comment count when available |
| 💬 Comment collection | Fetches public top-level comments through the YouTube Data API |
| 🧠 Sentiment prediction | Classifies each fetched comment as Positive, Negative, or Neutral |
| 📊 Summary cards | Shows the number and percentage of comments in each sentiment group |
| 🥧 Sentiment distribution | Interactive pie chart of sentiment percentages |
| 📊 Sentiment counts | Bar chart comparing the number of comments per class |
| 📈 Sentiment over time | Shows daily comment counts by sentiment when comment timestamps are available |
| ☁️ Word cloud | Highlights frequently used words in the fetched comments |
| 🔎 Search and filter | Search comment text and filter the table by sentiment |
| ⭐ Comment highlights | Shows highly liked positive and negative comments |
| 📥 CSV export | Downloads analyzed comments and predictions as a CSV file |

## 🚀 Live Demo

- **Live app:** Add your deployed Streamlit URL here, for example `https://your-app-name.streamlit.app`
- **GitHub repository:** Add your repository URL here.

> Replace the example links above after you deploy the app.

## 🖼️ Screenshots

Add screenshots of your running app to a folder named `screenshots/`, then update the image paths below.

```markdown
![Dashboard overview](screenshots/dashboard.png)
![Sentiment charts](screenshots/sentiment-charts.png)
![Analyzed comments](screenshots/analyzed-comments.png)
```

## 🧰 Tech Stack

- **Python** for application logic
- **Streamlit** for the interactive web app
- **Pandas and NumPy** for data handling
- **Scikit-learn** for the saved machine-learning pipeline
- **TF-IDF** for converting text into numeric features
- **Logistic Regression** for sentiment classification
- **Plotly** for interactive charts
- **Matplotlib and WordCloud** for the word cloud
- **YouTube Data API v3** for public video details and comments
- **Joblib** for loading the saved model

## 🧠 Machine Learning Pipeline

The app loads `sentiment_pipeline.pkl` and calls `model.predict()` on the fetched comment text.

```text
YouTube video URL
        |
        v
Extract video ID
        |
        v
YouTube Data API v3
        |
        v
Fetch public top-level comments
        |
        v
Saved TF-IDF + Logistic Regression pipeline
        |
        v
Positive / Negative / Neutral labels
        |
        v
Charts, word cloud, comment table, CSV
```

The saved pipeline includes the text vectorizer and classifier, so the app should pass raw comment text to the pipeline rather than manually applying a separate vectorizer.

## 📁 Project Structure

```text
youtube-sentiment-analysis/
├── app.py
├── sentiment_pipeline.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

Optional local secrets file:

```text
.streamlit/
└── secrets.toml
```

**Never commit `.streamlit/secrets.toml` or any file containing your API key.**

## ⚙️ Requirements

- Python version compatible with your installed dependencies
- A Google account to create a YouTube Data API key
- GitHub account for deployment
- The trained `sentiment_pipeline.pkl` file

### `requirements.txt`

Use the dependencies needed by the current app:

```text
streamlit
pandas
numpy
plotly
matplotlib
wordcloud
google-api-python-client
joblib
scikit-learn==1.9.1
```

The model was saved with scikit-learn 1.9.1. Use a compatible Python environment and matching scikit-learn version where possible. If pip cannot install this version for your Python version, resolve that compatibility issue before deploying; do not blindly change the version used to load the pickle.

## 💻 Run Locally

### 1. Clone your repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/youtube-sentiment-analysis.git
cd youtube-sentiment-analysis
```

Or download the repository as a ZIP file and open the folder in VS Code.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add your trained model

Place `sentiment_pipeline.pkl` in the same directory as `app.py`.

### 5. Configure the YouTube API key

Create a Google Cloud project, enable **YouTube Data API v3**, and create an API key:

- [Google Cloud Console](https://console.cloud.google.com/)
- [YouTube Data API documentation](https://developers.google.com/youtube/v3/docs/commentThreads/list)

For local development, create `.streamlit/secrets.toml`:

```toml
YOUTUBE_API_KEY = "YOUR_ACTUAL_API_KEY"
```

Create the `.streamlit` folder if it does not exist.

Alternatively, the app allows you to enter the key in its sidebar. Keep the key private and never commit it to Git.

### 6. Start Streamlit

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

### 7. Analyze a video

1. Enter your YouTube API key in the sidebar if it is not configured in secrets.
2. Paste a YouTube video URL or video ID.
3. Choose how many comments to fetch.
4. Choose newest-first or most-relevant ordering.
5. Click **Analyze Video**.
6. Explore the charts and download the CSV.

## 🔑 Create a YouTube API Key

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project.
3. Open **APIs & Services → Library**.
4. Enable **YouTube Data API v3**.
5. Open **APIs & Services → Credentials**.
6. Create an API key.
7. Restrict the key where practical and monitor its usage.

The API may not return comments for every video. Comments may be disabled, unavailable, or restricted. API quota limits also apply.

## ☁️ Deploy on Streamlit Community Cloud

### 1. Push the project to GitHub

Ensure the repository contains `app.py`, `requirements.txt`, `sentiment_pipeline.pkl`, and this README.

Do not upload API keys, `.env` files, or `.streamlit/secrets.toml`.

### 2. Open Streamlit Community Cloud

Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.

### 3. Create the app

- Select your repository.
- Choose the branch, usually `main`.
- Set the main file path to `app.py`.
- Choose a compatible Python version if the deployment settings offer that option.

### 4. Add the API key to Secrets

Open the app's **Advanced settings → Secrets** and add:

```toml
YOUTUBE_API_KEY = "YOUR_ACTUAL_API_KEY"
```

Save the settings and deploy.

### 5. Test the public app

After deployment completes, open the provided URL and test a public video whose comments are enabled. Confirm that model loading, comment retrieval, charts, and CSV export all work.

Official guide: [Deploy your app on Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).


## 🗺️ Future Improvements

- Add replies to top-level comments
- Add confidence scores if supported by the trained model
- Add filters by date, likes, and sentiment
- Improve text cleaning and multilingual support
- Add model evaluation metrics and a labeled test dataset
- Cache repeated video requests where appropriate
- Add a comparison view for multiple videos

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a branch for your change.
3. Make and test your changes.
4. Open a pull request describing what changed.

## 📄 License

Add a license before distributing this project. If you choose an open-source license, include its full license file in the repository.




'''

output_path = "/mnt/data/README.md"
pypandoc.convert_text(readme, "md", format="md", outputfile=output_path, extra_args=["--standalone"])
print(f"Created {output_path}")
