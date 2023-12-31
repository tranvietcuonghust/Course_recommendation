'''
Đọc vào 4 file raw
Xử lý: 
    level_mapping: chuẩn hóa level
    categories_mapping : chuẩn hóa categories, master_categories (lấy mapping chuẩn từ course_injection)
    Chuẩn hóa duration
    Chuẩn hóa price
    Xử lý null
    Xử lý provider, instructor
    Đặt lại cho đúng tên cột
Xử lý skills dùng cho Pluralsight và Futurelearn:
    ko thể chạy lại vì chatgpt
    thay thế bằng keybert
Bỏ phần data integrating ở cuối
=> input 4 file raw , output 4 file processed

** Course_integration: Đưa phần tách intructor link và instructor name/ provider về 4 file xử lý propresing
** Course_recommendation: Xử lý nlp ( lemmatiztion) và tách keyword => đưa về 4 file preprocessing

'''
import pandas as pd
import re
import ast
import numpy as np
import math
import spacy
from keybert import KeyBERT
from tqdm import tqdm
from spacy.cli.download import download

if "en_core_web_sm" not in spacy.util.get_installed_models():
    print("Downloading en_core_web_sm...")
    download("en_core_web_sm")
else:
    print("en_core_web_sm is already installed.")

pluralsight = pd.read_csv('./pluralsight_test5.csv')
processed_pluralsight = pluralsight.copy()

# Duration
def extract_rounded_hours(duration):
    if isinstance(duration, (str, int, float, np.float64)):
        duration_str = str(duration)
        match = re.search(r'(\d+)h (\d+)m', duration_str)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            total_hours = hours + math.ceil(minutes / 60)
            return total_hours
    return None

processed_pluralsight["Duration"] = processed_pluralsight['Duration'].apply(extract_rounded_hours)
processed_pluralsight["Duration"] = processed_pluralsight["Duration"].fillna(0).astype(int)


# Instructor
def extract_instructor_name(instructor_info):
    try:
        instructor_dict = ast.literal_eval(instructor_info)
        return instructor_dict['instructor_name']
    except (ValueError, KeyError):
        return None
processed_pluralsight['InstructorName'] = processed_pluralsight['Instructor'].apply(extract_instructor_name)
processed_pluralsight['InstructorName'] = processed_pluralsight['InstructorName'].str.replace('\n', '')
processed_pluralsight = processed_pluralsight.rename(columns={'Instructor': 'Instructor_full'})



# level
level_mapping = {'Novice': 'Beginner', 'Practitioner': 'Beginner', 'Professional': 'Advanced', 'Introductory':'Beginner','Apprentice':'Beginner','Guru':'Advanced', 'Beginner':'Beginner','Intermediate':'Intermediate','Advanced':'Advanced'}

processed_pluralsight['Level'] = processed_pluralsight['Level'].str.replace('\n', '')
processed_pluralsight['Level'] = processed_pluralsight['Level'].replace({k: v for k, v in level_mapping.items()})



# processed_pluralsight = processed_pluralsight.drop(columns=['Instructor','NumReviews'])



# categories
categories_keywords = {
    'Computer Science': ['computer science', 'algorithms', 'software development', 'mobile development'],
    'Computer Security and Networks': ['computer security', 'networks'],
    'Data Science': ['data science', 'machine learning', 'data analysis', 'statistics', 'big data', 'analytics'],
    'Business': ['business', 'business strategy', 'leadership', 'management', 'marketing', 'finance', 'entrepreneurship'],
    'Business Strategy': ['business strategy'],
    'Leadership and Management': ['leadership', 'management'],
    'Information Technology': ['information technology', 'support', 'operations', 'cloud computing', 'data management', 'networking', 'security'],
    'Arts and Humanities': ['arts', 'humanities', 'history', 'music', 'art', 'philosophy', 'ethics', 'literature', 'architecture', 'creative arts', 'media'],
    'Personal Development': ['personal development', 'communication'],
    'Physical Science and Engineering': ['physical science', 'engineering', 'mechanical engineering', 'environmental science', 'sustainability', 'physics', 'astronomy', 'electrical engineering', 'electronic', 'chemistry', 'research methods', 'energy', 'earth sciences'],
    'Health': ['health', 'basic science', 'animal health', 'patient care', 'psychology', 'nutrition', 'health informatics', 'public health', 'healthcare management', 'medicine'],
    'Social Sciences': ['social sciences', 'law', 'education', 'teaching', 'study skills', 'economics', 'governance', 'society', 'philanthropy', 'politics'],
    'Language Learning': ['language learning', 'learning English', 'other languages'],
    'Math and Logic': ['math', 'mathematics', 'logic'],
    'Biology & Life Sciences': ['biology', 'life sciences'],
    'Nature & Environment': ['nature', 'environment'],
    'Science, Engineering & Maths': ['science', 'engineering', 'maths'],
    'Uncategorized': []
}

def assign_category(description):
    for category, keywords in categories_keywords.items():
        if any(keyword.lower() in description.lower() for keyword in keywords):
            return category
    return 'Uncategorized'
processed_pluralsight['Categories'] = processed_pluralsight['Descriptions'].apply(assign_category)

processed_pluralsight['Categories'].fillna('Uncategorized', inplace=True)

Categories_mapping ={
    'Design' : 'Design and Product',
    'Teaching': 'Education',
    'Study Skills':'Education',
    'Computer Science': 'IT & Computer Science',
    'Data Analysis': 'Data Analysis & Statistics',
    'Business': 'Business & Management',
    'Music': 'Music and Art',
    'Philosophy':'Philosophy & Ethics',
    'Ethics' :'Philosophy & Ethics',
    'Environmental Science and Sustainability': 'Environmental Studies',
    'Health':'Health & Safety',
    'Nutrition':'Food & Nutrition',
    'Psychology':'Psychology & Mental Health',
    'Medicine':'Healthcare & Medicine',
    'Education & Teacher Training': 'Education',
    'Language Learning' :'Language',
    'Math and Logic': 'Mathematics & Logic',
    'Basic Science' : 'Science',
    'Research':'Research Methods'
      # Default category if no match
}

processed_pluralsight['Categories'] = processed_pluralsight['Categories'].map(Categories_mapping).fillna(processed_pluralsight['Categories'])

Master_categories_mapping ={
    'Computer Science': 'IT & Computer Science',
    'Computer Security and Networks': 'IT & Computer Science',
    'Algorithms': 'IT & Computer Science',
    'Design and Product': 'IT & Computer Science',
    'Design': 'IT & Computer Science',
    'Software Development': 'IT & Computer Science',
    'Mobile and Web Development': 'IT & Computer Science',
    'IT & Computer Science': 'IT & Computer Science',

    'Data Science': 'Data Science',
    'Machine Learning': 'Data Science',
    'Data Analysis': 'Data Science',
    'Data Analysis & Statistics': 'Data Science',
    'Probability and Statistics': 'Data Science',
    'Big Data & Analytics ': 'Data Science',

    'Business': 'Business & Management',
    'Business & Management': 'Business & Management',
    'Business Strategy': 'Business & Management',
    'Leadership and Management': 'Business & Management',
    'Marketing': 'Business & Management',
    'Finance': 'Business & Management',
    'Business Essentials': 'Business & Management',
    'Entrepreneurship': 'Business & Management',
    'Economics & Finance': 'Business & Management',

    'Information Technology': 'Information Technology',
    'Support and Operations': 'Information Technology',
    'Cloud Computing': 'Information Technology',
    'Data Management': 'Information Technology',
    'Networking': 'Information Technology',
    'Security': 'Information Technology',

    'Arts and Humanities': 'Arts and Humanities',
    'History': 'Arts and Humanities',
    'Music and Art': 'Arts and Humanities',
    'Music': 'Arts and Humanities',
    'Philosophy': 'Arts and Humanities',
    'Philosophy & Ethics': 'Arts and Humanities',
    'Ethics': 'Arts and Humanities',
    'Humanities': 'Arts and Humanities',
    'Literature': 'Arts and Humanities',
    'Art & Culture': 'Arts and Humanities',
    'Architecture': 'Arts and Humanities',
    'Creative Arts & Media': 'Arts and Humanities',

    'Personal Development': 'Personal Development',
    'Communication': 'Personal Development',

    'Physical Science and Engineering': 'Physical Science and Engineering',
    'Mechanical Engineering': 'Physical Science and Engineering',
    'Environmental Science and Sustainability': 'Physical Science and Engineering',
    'Environmental Studies': 'Physical Science and Engineering',
    'Physics and Astronomy': 'Physical Science and Engineering',
    'Electrical Engineering': 'Physical Science and Engineering',
    'Electronics': 'Physical Science and Engineering',
    'Chemistry': 'Physical Science and Engineering',
    'Research Methods': 'Physical Science and Engineering',
    'Physics': 'Physical Science and Engineering',
    'Engineering': 'Physical Science and Engineering',
    'Energy & Earth Sciences': 'Physical Science and Engineering',
    'Basic Science': 'Physical Science and Engineering',
    'Science': 'Physical Science and Engineering',

    'Health': 'Health & Safety',
    'Health & Safety': 'Health & Safety',
    'Healthcare & Medicine': 'Health & Safety',
    'Animal Health': 'Health & Safety',
    'Patient Care': 'Health & Safety',
    'Psychology': 'Health & Safety',
    'Psychology & Mental Health': 'Health & Safety',
    'Nutrition': 'Health & Safety',
    'Food & Nutrition': 'Health & Safety',
    'Health Informatics': 'Health & Safety',
    'Public Health': 'Health & Safety',
    'Research': 'Health & Safety',
    'Healthcare Management': 'Health & Safety',
    'Medicine': 'Health & Safety',

    'Social Sciences': 'Social Sciences',
    'Law': 'Social Sciences',
    'Education': 'Social Sciences',
    'Education & Teacher Training': 'Social Sciences',
    'Teaching': 'Social Sciences',
    'Study Skills': 'Social Sciences',
    'Economics': 'Social Sciences',
    'Governance and Society': 'Social Sciences',
    'Philanthropy': 'Social Sciences',
    'Politics & Society': 'Social Sciences',

    'Language Learning': 'Language',
    'Language': 'Language',
    'Learning English': 'Language',
    'Other Languages': 'Language',

    'Math and Logic': 'Mathematics & Logic',
    'Mathematics & Logic': 'Mathematics & Logic',
    'Math': 'Mathematics & Logic',

    'Biology & Life Sciences': 'Biology & Life Sciences',

    'Nature & Environment': 'Nature & Environment',
    'Science, Engineering & Maths' :'Uncategorized',

    'Uncategorized': 'Uncategorized'  # Default category if no match
}

processed_pluralsight['MasterCategories'] = ""
processed_pluralsight['MasterCategories'] = processed_pluralsight['Categories'].map(Master_categories_mapping).fillna(processed_pluralsight['MasterCategories'])

# provider
processed_pluralsight['ProviderName'] = ''
processed_pluralsight['Provider_full'] = [{'provider_link': None, 'provider_name': None}] * len(processed_pluralsight)

# Modules
processed_pluralsight['Modules'] = processed_pluralsight['Modules'].str.replace('\n', '')
processed_pluralsight['Modules'].fillna('[{"module_number": None, "module_name": None, "module_duration": None}]', inplace=True)

# NumEnrolled
processed_pluralsight['NumEnrolled'] = 'None'

# NumReviews
processed_pluralsight['NumReviews'] = processed_pluralsight['NumEnrolled']

# ReviewsURL
processed_pluralsight['ReviewsURL'] = ''

# Stars
processed_pluralsight['Stars'].replace(to_replace=0, value=5.0, inplace=True)

# Skills
# processed_pluralsight.rename(columns={'Summarized_Skills': 'Skills'}, inplace=True)
file2 = './merged_course_with_keywords.csv'
df2 = pd.read_csv(file2)
# Merge DataFrame theo cột 'Link'
processed_pluralsight = pd.merge(processed_pluralsight, df2[['Link', 'Skills']], on='Link', how='left')

# Source
processed_pluralsight['Source'] = 'Pluralsight'

# string
processed_pluralsight['string'] = processed_pluralsight['Name'].str.cat(processed_pluralsight['Descriptions'], sep=' ')
processed_pluralsight['string'] = processed_pluralsight['string'].astype(str)

# Load the English NLP model from SpaCy
nlp = spacy.load('en_core_web_sm')

def apply_lemmatization(text):
    # Process the text using SpaCy
    doc = nlp(text)

    # Extract lemmatized tokens
    lemmatized_tokens = [token.lemma_ for token in doc]

    # Join the lemmatized tokens back into a single string
    lemmatized_text = ' '.join(lemmatized_tokens)

    return lemmatized_text

processed_pluralsight['string']=processed_pluralsight['string'].apply(apply_lemmatization)

model = KeyBERT()

# Extract top 10 keywords for each row in the 'Merged' column
tqdm.pandas()
processed_pluralsight['Keywords'] = processed_pluralsight['string'].progress_apply(lambda x: ', '.join(keyword[0] for keyword in model.extract_keywords(x, top_n=10, diversity = 0.8, use_mmr =True)))

processed_pluralsight['Keywords'] = processed_pluralsight['Keywords'].str.cat(processed_pluralsight['MasterCategories'], sep=', ')
processed_pluralsight['Keywords'] = processed_pluralsight['Keywords'].str.cat(processed_pluralsight['Categories'], sep=', ')
processed_pluralsight['Keywords'] = processed_pluralsight['Keywords'].str.cat(processed_pluralsight['Skills'], sep=', ')
processed_pluralsight['Keywords'] = processed_pluralsight['Keywords'].astype(str)


# Ghi DataFrame đã xử lý vào file CSV
processed_pluralsight.to_csv('./data/pluralsight_processed.csv', index=False, mode='w')

# In ra các cột đã sắp xếp
print(sorted(processed_pluralsight.columns))