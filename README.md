# Amazon Reviews Sentiment Analysis

A machine learning pipeline to classify Amazon product reviews into positive or negative sentiments using TF-IDF and Logistic Regression.

## Project Structure
- `src/`: Core python scripts for cleaning and training.
- `models/`: Saved model and vectorizer binaries (`.pkl`).
- `notebooks/`: Exploratory data analysis notebooks.

## Model Performance
- **Accuracy:** ~93%
- Handles class imbalance using `class_weight='balanced'`.