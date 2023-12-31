'''
Đọc vào 4 file processed
Tách 5 cột chung và cột link
Gộp lại thành 1 dataframe (lưu ý tên cột phải giống nhau)
Phân cụm và gán nhãn cụm cho từng course
Lưu lại thành 1 file merged_course (nhưng thiếu cột)
'''
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from tqdm import tqdm
import re

processed_coursera = pd.read_csv('./data/coursera_processed.csv')
processed_edx = pd.read_csv('./data/edx_processed.csv')
processed_pluralsight = pd.read_csv('./data/pluralsight_processed.csv')
processed_futurelearn = pd.read_csv('./data/futurelearn_processed.csv')

# columns_to_extract = ['Link', 'Name', 'Hours', 'Categories', 'MasterCategories', 'Descriptions', 'InstructorName','Skills']

# coursera_subset = processed_coursera[columns_to_extract]
# edx_subset = processed_edx[columns_to_extract]
# pluralsight_subset = processed_pluralsight[columns_to_extract]
# futurelearn_subset = processed_futurelearn[columns_to_extract]

merged_course = pd.concat([processed_coursera, processed_edx, processed_pluralsight, processed_futurelearn], ignore_index=True)

merged_course.fillna("", inplace=True)
merged_course['tokenized_description'] = merged_course['Descriptions'].apply(word_tokenize)

# Count word frequencies
all_words = [word for tokens in merged_course['tokenized_description'] for word in tokens]
fdist = FreqDist(all_words)

#Set a frequency threshold (adjust as needed)
frequency_threshold = 2400

# Identify stop words based on the threshold
description_stop_words = {word for word, freq in fdist.items() if freq >= frequency_threshold}

# NLTK setup
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

# Update your stop_word set
stop_words.update(description_stop_words)

clean_descriptions = []

for description in tqdm(merged_course['Descriptions'], colour='yellow'):
    if isinstance(description, str):  # Check if description is a string
        # Remove all except the alphabets
        description = re.sub("[^a-zA-Z]", ' ', description)

        # Lower all the alphabets
        description = description.lower()

        # Split the description on spaces, returning a list of words
        words = description.split()

        # Remove stopwords and apply stemming
        clean_description = [word for word in words if word not in stop_words]

        # Join clean words
        clean_description = " ".join(clean_description)

        # Append the cleaned description
        clean_descriptions.append(clean_description)
    else:
        clean_descriptions.append("")  # If description is not a string, append an empty string

# Update the DataFrame with cleaned descriptions
merged_course['Clean_Description'] = clean_descriptions

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Assuming 'Skills' is a list of strings, we convert it to a string representation
# merged_course['Skills'] = merged_course['Skills'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x).str.replace('[', '').replace(']','')
merged_course.fillna("", inplace=True)
# # TF-IDF Vectorization for each column
vectorizer_name = TfidfVectorizer()
tfidf_matrix_name = vectorizer_name.fit_transform(merged_course['Name'])

vectorizer_categories = TfidfVectorizer()
tfidf_matrix_categories = vectorizer_categories.fit_transform(merged_course['Categories'])

vectorizer_master_categories = TfidfVectorizer()
tfidf_matrix_master_categories = vectorizer_master_categories.fit_transform(merged_course['MasterCategories'])

vectorizer_description = TfidfVectorizer()
tfidf_matrix_description = vectorizer_description.fit_transform(merged_course['Clean_Description'])

vectorizer_skills = TfidfVectorizer()
tfidf_matrix_skills = vectorizer_skills.fit_transform(merged_course['Skills'])
# Use cosine similarity for each column
similarity_name = cosine_similarity(tfidf_matrix_name)
similarity_categories = cosine_similarity(tfidf_matrix_categories)
similarity_master_categories = cosine_similarity(tfidf_matrix_master_categories)
similarity_description = cosine_similarity(tfidf_matrix_description)
similarity_skills = cosine_similarity(tfidf_matrix_skills)

# Define weights for each column
weights = { 'Name':0.2, 'Clean_Description':0.2, 'Categories': 0.2, 'MasterCategories': 0.2
            , 'Skills': 0.2
           }

# Calculate weighted similarity for each column
weighted_similarity = (
    weights['Name'] * similarity_name +
    weights['Categories'] * similarity_categories +
    weights['MasterCategories'] * similarity_master_categories +
    weights['Clean_Description'] * similarity_description +
    weights['Skills'] * similarity_skills
)

print(weighted_similarity)

from scipy.cluster.hierarchy import linkage, fcluster
linkage_matrix = linkage(weighted_similarity, method='complete')  # You can choose a different linkage method based on your requirements

# Determine clusters based on distance or a specified number of clusters
# For example, you can choose a distance threshold:
distance_threshold =  1.0  # Adjust based on your dendrogram
clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')

merged_course['Cluster'] = clusters

merged_course = merged_course.drop(['tokenized_description', 'Clean_Description'], axis=1)

merged_course.to_csv('./data/merged_course_final.csv', index=False)