---

# 📝 Sentiment Analysis Web Application

This project is a Flask-based web application that performs sentiment analysis on restaurant reviews. It leverages Natural Language Processing (NLP) techniques and a Random Forest Classifier to predict whether a given review is positive or negative.

---

## 🚀 Features

* **User-Friendly Interface**: Input your review through a simple web form.
* **Real-Time Prediction**: Get immediate feedback on the sentiment of your review.
* **Data Preprocessing**: Cleans and processes text data using NLTK.
* **Machine Learning Model**: Utilizes a Random Forest Classifier for accurate predictions.
* **Performance Metrics**: Displays accuracy, precision, recall, and confusion matrix.([CodeMax][1])

---

## 🧠 Technologies Used

* **Python**: Core programming language.
* **Flask**: Web framework for creating the application.
* **Pandas & NumPy**: Data manipulation and numerical operations.
* **NLTK**: Text preprocessing and NLP tasks.
* **Scikit-learn**: Machine learning algorithms and evaluation metrics.
* **Matplotlib & Seaborn**: Data visualization.([GitHub][2])

---

## 🗂️ Project Structure

```

├── app.py
├── Restaurant_Reviews.tsv
├── templates/
│   └── part2.html
├── static/
│   └── (Optional static files like CSS or JS)
├── requirements.txt
└── README.md
```



---

## 📦 Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/VanshGupta-2404/Sentiment_analysis.git
   cd Sentiment_analysis
   ```



2. **Create a Virtual Environment** (Optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



---

## 🏃‍♂️ Running the Application

1. **Start the Flask App**:

   ```bash
   python app.py
   ```



2. **Access the Application**:
   Open your browser and navigate to `http://127.0.0.1:5000/`

---

## 🧪 How It Works

1. **Data Loading**:
   Reads the `Restaurant_Reviews.tsv` file containing reviews and their corresponding sentiments.

2. **Data Preprocessing**:

   * Removes non-alphabetic characters.
   * Converts text to lowercase.
   * Removes stopwords.
   * Applies stemming using PorterStemmer.

3. **Feature Extraction**:
   Transforms text data into numerical features using TF-IDF Vectorization.

4. **Model Training**:
   Splits the data into training and testing sets and trains a Random Forest Classifier.

5. **Evaluation**:
   Calculates accuracy, precision, recall, and displays a confusion matrix.

6. **Prediction**:
   Takes user input from the web form, preprocesses it, and predicts the sentiment.

---

## 📈 Performance Metrics

* **Accuracy**: \~85% (varies based on dataset and parameters)
* **Precision**: \~84%
* **Recall**: \~86%

*Note: These metrics are illustrative; actual results may vary.*

---

## 🔧 Customization

* **Model Parameters**: Adjust the number of estimators in the Random Forest Classifier for better performance.
* **Preprocessing**: Modify stopword lists or stemming techniques as needed.
* **UI Enhancements**: Customize `part2.html` for a better user interface.

---

## 📬 Contact

For any queries or suggestions, feel free to reach out:

* **Email**: [guptavansh2404@gmail.com](mailto:guptavansh2404@gmail.com)
* **GitHub**: [VanshGupta-2404](https://github.com/VanshGupta-2404)

---
