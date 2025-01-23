**Sentiment Analysis Project Overview**  
This project aims to address the challenge of aggregating and interpreting user feedback effectively, specifically focusing on Yelp reviews. With user feedback being a key metric for product evaluation, the project introduces a natural language processing (NLP) pipeline to classify sentiments expressed in reviews quickly and accurately. The process involves scraping reviews from the Apple App Store, preprocessing text data to standardize its format, and using exploratory data analysis (EDA) to uncover patterns in language usage across review categories. Word clouds and unique word counts helped highlight differences in extreme and neutral contexts, emphasizing the varied nature of user feedback.

**Exploratory Data Analysis**
The exploratory data analysis involved developing several functions that would preprocess the text and remove common stop words, as well as a function to examine the word counts of unique words in each category. Below is one figure that had a key finding:
[]
Figure 1. Word counts for unique words in each category, showing frequently used words in positive, neutral, and negative contexts

**Model Development and Performance**  
The project experimented with various NLP modeling techniques, starting with random forest classifiers using different text vectorization methods, including CountVectorizer, TfidfVectorizer, and N-grams. Initial models achieved moderate accuracy (up to 43%) but revealed room for improvement, particularly in handling multiclass tasks. A shift to word embeddings with the Gensim package enabled the training of a neural network that better captured contextual relationships. This enhancement, along with increased training data (20,000 reviews), resulted in a binary sentiment classification model with an accuracy of 92%. Multiclass predictions saw a slight improvement as well, with the best model showing an accuracy of 48%, and displayed good performance when predicting extremes (with 1-star and 5-star review categories showing F1-scores of 0.61 and 0.66), but struggled to accurately predict more neutral cases (F1-score of 0.34 for 3-star reviews). This suggests that the neutral reviews contain noisy data, and is supported by the results from the EDA. In order to find the model with the least overfitting, gridsearch was used and found the best model with the least overfitting was a logistic regression model with an L2 penalty and a C value of 0.1.

[]
Figure 2. Binary classification results confusion matrix

[]
Figure 3. Multiclass classification results confusion matrix using most optimized model

**Key Insights, Limitations, and Future Directions**  
The findings highlight the strength of binary sentiment classification, particularly for extreme sentiments, and the promise of word embedding-based models for improved performance. However, limitations include the noisiness of neutral data and the inability to process words absent from the training corpus, which points to the need for broader word embedding models. To enhance the system's robustness, future steps include collecting more data, refining the word embedding schemes, and exploring alternative feature extraction methods. These developments will further enable developers to understand user sentiment and improve their products effectively.
