
import joblib

# Load your existing trained pipeline
model = joblib.load("sentiment_pipeline.pkl")

# Test comments
comments = [
    "This video is amazing and very helpful!",
    "This video is terrible and a waste of time.",
    "The video was uploaded yesterday."
]

# Predict sentiment
predictions = model.predict(comments)

# Display results
for comment, sentiment in zip(comments, predictions):
    print("Comment:", comment)
    print("Sentiment:", sentiment)
    print("-" * 50)
