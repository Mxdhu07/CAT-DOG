import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained model and vectorizer
with open('clf.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)

with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

# Function to normalize text
def normalize_text(text):
    text = re.sub(r'\bHuman Resources\b', 'HR', text, flags=re.IGNORECASE)
    text = re.sub(r'\bSoftware Engineer\b', 'SE', text, flags=re.IGNORECASE)

    return text

# Function to preprocess text
def preprocess_text(text):
    text = normalize_text(text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to classify a resume
def classify_resume(text):
    cleaned_resume = preprocess_text(text)
    input_features = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]

    specific_to_broad_category = {
        "Java Developer": "Software Engineering",
        "Testing": "Quality Assurance",
        "DevOps Engineer": "Engineering",
        "Python Developer": "Software Engineering",
        "Web Designing": "Design",
        "HR": "Human Resources",
        "Hadoop": "Data Engineering",
        "Blockchain": "Engineering",
        "ETL Developer": "Data Engineering",
        "Operations Manager": "sales",
        "Data Science": "Data Science",
        "Sales": "Sales",
        "Mechanical Engineer": "Engineering",
        "Arts": "Arts",
        "Database": "Database Administration",
        "Electrical Engineering": "Engineering",
        "Health and Fitness": "Medical",
        "PMO": "Project Management",
        "Business Analyst": "Business Analysis",
        "Advocate": "Law",
        "DotNet Developer": "Software Engineering",
        "Automation Testing": "Quality Assurance",
        "Network Security Engineer": "Engineering",
        "SAP Developer": "Software Engineering",
        "Civil Engineer": "Engineering"
    }

    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and Fitness",
        19: "PMO",
        4: "Business Analyst",
        0: "Advocate",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer"
    }

    specific_category = category_mapping.get(prediction_id, "Unknown")
    broad_category = specific_to_broad_category.get(specific_category, "Unknown")
    return broad_category
