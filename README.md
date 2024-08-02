### SPAM SMS Detection

#### Overview
This project aims to detect spam messages from a dataset of SMS messages using various machine learning techniques. The project involves data preprocessing, feature extraction, model training, evaluation, and visualization of results.

#### Steps Involved

1. **Loading the Dataset**
   - Load the dataset containing SMS messages labeled as 'spam' or 'ham' (non-spam).
   - Dataset Source: Provided CSV file.

2. **Data Preprocessing**
   - Handle missing values by filling them appropriately.
   - Convert all text to lowercase to ensure uniformity.
   - Remove punctuation and stop words from the text to clean the data.
   - Tokenize the text into words.

3. **Feature Extraction**
   - Convert text data into numerical features using techniques such as TF-IDF (Term Frequency-Inverse Document Frequency).
   - Generate n-grams to capture the context of words in the messages.

4. **Model Training**
   - Split the dataset into training and testing sets.
   - Train various machine learning models such as Naive Bayes, Support Vector Machine (SVM), and Random Forest.
   - Optimize model parameters using techniques such as Grid Search or Random Search.

5. **Model Evaluation**
   - Evaluate the trained models using metrics like accuracy, precision, recall, and F1-score.
   - Compare the performance of different models and select the best-performing model.

6. **Visualization**
   - Visualize the distribution of spam and ham messages in the dataset.
   - Plot confusion matrices for the models to understand their performance better.
   - Visualize the most important features contributing to spam detection.

#### Files

- **spam_sms_dataset.csv**: The dataset containing SMS messages and their labels.
- **spam_detection.ipynb**: Jupyter Notebook with code, explanations, and results of the analysis.
- **README.md**: Project overview and instructions for running the code.

#### Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk (Natural Language Toolkit)
