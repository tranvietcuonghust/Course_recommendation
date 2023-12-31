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
import spacy
from keybert import KeyBERT
from tqdm import tqdm
from spacy.cli.download import download

if "en_core_web_sm" not in spacy.util.get_installed_models():
    print("Downloading en_core_web_sm...")
    download("en_core_web_sm")
else:
    print("en_core_web_sm is already installed.")

edx = pd.read_csv('./edx_test9.csv')
processed_edx=edx.copy()

# Duration
def extract_total_hours(duration):
    numbers = re.findall(r'\d+', duration)
    if len(numbers) >= 3:
        return int(numbers[0]) * int(numbers[2])
    else:
        return None

processed_edx['Duration'] = processed_edx['Duration'].apply(extract_total_hours)

# Instructor_name error (bi loi vi khong co name)
def extract_instructor_name(instructor_info):
    try:
        instructor_dict = ast.literal_eval(instructor_info)
        return instructor_dict['instructor_name']
    except (ValueError, KeyError):
        return None
    
processed_edx['InstructorName'] = processed_edx['Instructor'].apply(extract_instructor_name)
processed_edx = processed_edx.rename(columns={'Instructor': 'Instructor_full'})

# Price
def extract_price_number(price):
    if isinstance(price, (str, int, float, np.float64)):
        numbers = re.findall(r'\d+', str(price))
        if numbers:
            return ''.join(numbers)
    return None

processed_edx['Price'] = processed_edx['Price'].apply(extract_price_number)

# Provider_name
def extract_provider_name(provider_info):
    try:
        provider_dict = ast.literal_eval(provider_info)
        return provider_dict['provider_name']
    except (ValueError, KeyError):
        return None
    
processed_edx['ProviderName'] = processed_edx['Provider'].apply(extract_provider_name)
processed_edx = processed_edx.rename(columns={'Provider': 'Provider_full'})

# level
level_mapping = {'Novice': 'Beginner', 'Practitioner': 'Beginner', 'Professional': 'Advanced', 'Introductory':'Beginner','Apprentice':'Beginner','Guru':'Advanced', 'Beginner':'Beginner','Intermediate':'Intermediate','Advanced':'Advanced'}

processed_edx['Level'] = processed_edx['Level'].str.extract(r'<!-- -->(.+?)</li>')
processed_edx['Level'] = processed_edx['Level'].replace({k: v for k, v in level_mapping.items()})

# NumEnrolled
processed_edx['NumEnrolled'] = processed_edx['NumEnrolled'].str.extract(r'([\d,]+)')

processed_edx = processed_edx.drop(columns=['ImgURL'])

# categories
processed_edx['Categories'].fillna('Uncategorized', inplace=True)

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

processed_edx['Categories'] = processed_edx['Categories'].map(Categories_mapping).fillna(processed_edx['Categories'])

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

processed_edx['MasterCategories'] = ""
processed_edx['MasterCategories'] = processed_edx['Categories'].map(Master_categories_mapping).fillna(processed_edx['MasterCategories'])

# modules
processed_edx['Modules'] = '[{"module_number": None, "module_name": None, "module_duration": None}]'

# stars
processed_edx['Stars'] = '5.0'

# NumReviews
processed_edx['NumReviews'] = processed_edx['NumEnrolled']

# ReviewsURL
processed_edx['ReviewsURL'] = ''

# Source
processed_edx['Source'] = 'Edx'

# string
processed_edx['string'] = processed_edx['Name'].str.cat(processed_edx['Descriptions'], sep=' ')
processed_edx['string'] = processed_edx['string'].astype(str)

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

processed_edx['string']=processed_edx['string'].apply(apply_lemmatization)

model = KeyBERT()

# Extract top 10 keywords for each row in the 'Merged' column
tqdm.pandas()
processed_edx['Keywords'] = processed_edx['string'].progress_apply(lambda x: ', '.join(keyword[0] for keyword in model.extract_keywords(x, top_n=10, diversity = 0.8, use_mmr =True)))

processed_edx['Keywords'] = processed_edx['Keywords'].str.cat(processed_edx['MasterCategories'], sep=', ')
processed_edx['Keywords'] = processed_edx['Keywords'].str.cat(processed_edx['Categories'], sep=', ')
processed_edx['Keywords'] = processed_edx['Keywords'].str.cat(processed_edx['Skills'], sep=', ')
processed_edx['Keywords'] = processed_edx['Keywords'].astype(str)

# Ghi DataFrame đã xử lý vào file CSV
processed_edx.to_csv('./data/edx_processed.csv', index=False, mode='w')

# In ra các cột đã sắp xếp
print(sorted(processed_edx.columns))