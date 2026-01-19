# Sentiment Analysis of Text Using Naive Bayes

## 1. Objective

The main objective of this project is to design and implement a sentiment analysis model that:

1. Processes raw textual data

2. Extracts relevant textual features

3. Classifies text into Positive, Negative, or Neutral sentiments using a supervised learning approach

4. Evaluates the model using standard performance metrics

## 2. Dataset Description

A publicly available [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data) from Kaggle was used for this project.
The dataset consists of tweets labeled into the following sentiment categories:

- Positive

- Negative

- Neutral

Tweets labeled as Irrelevant were treated as Neutral, as they do not express sentiment toward the given entity. Missing values were removed during preprocessing to ensure data quality.

## 3. Tools and Technologies

- **Programming Language:** Python

- **Text Processing Library:** spaCy

- **Machine Learning Library:** scikit-learn

- **Feature Extraction Technique:** TF-IDF (Term Frequency–Inverse Document Frequency)

## 4. Text Preprocessing

Text preprocessing was performed using the spaCy NLP library. The following steps were applied:

1. Tokenization of text

2. Removal of stopwords

3. Removal of punctuation

4. Lemmatization to convert words into their base form

These steps help reduce noise in the text and improve the quality of features extracted for model training.

## 5. Feature Extraction

The TF-IDF vectorization technique was used to convert textual data into numerical form.
TF-IDF assigns weights to words based on:

- Their frequency within a document

- Their rarity across the entire dataset

This helps the classifier focus on informative words rather than common ones.

## 6. Model Training

A Multinomial Naïve Bayes classifier was used for sentiment classification.
The dataset was split into:

80% training data

20% testing data

A machine learning pipeline was created combining TF-IDF vectorization and the Naïve Bayes classifier to streamline training and prediction.

## 7. Model Evaluation

The performance of the trained model was evaluated using the following metrics:

- Accuracy

- Precision

- Recall

These metrics measure the model’s effectiveness in predicting sentiment labels on unseen data.

## 8. Testing on Unseen Data

The trained model was tested on new, unseen tweet text to verify its ability to correctly predict sentiment outside the training dataset.

## 9. Conclusion

This project demonstrates the practical application of NLP and machine learning in real-world text classification tasks.

The Naïve Bayes classifier achieved an overall **accuracy of 79%**, with weighted precision, recall, and F1-score values of approximately **0.79**, indicating balanced and reliable performance. The model performed particularly well in identifying neutral sentiment while maintaining high precision for positive and negative sentiment classes.
