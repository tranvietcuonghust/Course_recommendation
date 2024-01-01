'''
Mục tiêu: Đưa data về các bảng
Clean code lại cho đỡ lặp
'''
import pandas as pd
import psycopg2
# from sqlalchemy import create_engine
import ast

merged_course = pd.read_csv('./data/merged_course_final.csv')

# Replace these with your actual connection details
host = "c-course-inte.conr2igjpgjdaw.postgres.cosmos.azure.com"
dbname = "citus"
user = "citus"
password = "Lufe22022001"
sslmode = "require"

# Establish a connection to your PostgreSQL database
conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
conn = psycopg2.connect(conn_string)
cur = conn.cursor()


# Bảng Lessons
merged_course['Modules'] = merged_course['Modules'].apply(ast.literal_eval)
df_lessons = merged_course.explode('Modules')
df_lessons = df_lessons[['Link','Modules']]
df_lessons['ModuleNumber'] = df_lessons['Modules'].apply(lambda x: x['module_number'])
df_lessons['ModuleName'] = df_lessons['Modules'].apply(lambda x: x['module_name'])
df_lessons['ModuleDuration'] = df_lessons['Modules'].apply(lambda x: x['module_duration'])
df_lessons  = df_lessons[['Link','ModuleNumber','ModuleName','ModuleDuration']]
df_lessons= df_lessons.dropna(subset=["ModuleNumber","ModuleName"])
df_lessons['ModuleName'] = df_lessons['ModuleName'].str.replace('\n', '')
df_lessons['ModuleDuration'] = df_lessons['ModuleDuration'].str.replace('\n', '')
df_lessons=df_lessons.reset_index(drop=True)

# Define the Lessons table
table_name = "Lessons4"
cur.execute("""
    CREATE TABLE IF NOT EXISTS {0} (
        Lesson_id SERIAL PRIMARY KEY,
        Link VARCHAR(255) NOT NULL,
        ModuleNumber INT ,
        ModuleName VARCHAR(255) ,
        ModuleDuration VARCHAR(255)
    );
""".format(table_name))
conn.commit()

# Insert data into the Lessons table
for index, row in df_lessons.iterrows():
    cur.execute("""
        INSERT INTO {0} (Link, ModuleNumber, ModuleName, ModuleDuration)
        VALUES (%s, %s, %s, %s);
    """.format(table_name), (row['Link'], row['ModuleNumber'], row['ModuleName'], row['ModuleDuration']))
    conn.commit()


# Bảng CourseCluster
df_course_clusters = merged_course[["Cluster"]].copy()
df_course_clusters['Cluster_name'] = 'Cluster ' + df_course_clusters['Cluster'].astype(str)
df_course_clusters = df_course_clusters.drop_duplicates()

# Define the CourseCluster table
table_name = "course_cluster"
cur.execute("""
    CREATE TABLE IF NOT EXISTS {0} (
        cluster_id SERIAL PRIMARY KEY,
        Cluster VARCHAR(255),
        Cluster_name VARCHAR(255)
    );
""".format(table_name))
conn.commit()

# Insert data into the CourseCluster table
for index, row in df_course_clusters.iterrows():
    cur.execute("""
        INSERT INTO {0} (Cluster, Cluster_name)
        VALUES (%s, %s);
    """.format(table_name), (row['Cluster'],  row['Cluster_name']))
    conn.commit()


# Bảng Course_Skills
df_courses_skills = merged_course[["Link", "Skills"]].copy()
df_courses_skills["Skills"] = df_courses_skills["Skills"].str.split(",")
df_courses_skills = df_courses_skills.explode("Skills")
df_courses_skills = df_courses_skills.dropna(subset=["Skills"]).explode("Skills").reset_index(drop=True)

# Define the Course_Skills table
table_name = "course_skill"
cur.execute("""
    CREATE TABLE IF NOT EXISTS {0} (
        Course_skill_id SERIAL PRIMARY KEY,
        Link TEXT NOT NULL,
        Skill VARCHAR(255)
    );
""".format(table_name))
conn.commit()

# Insert data into the Course_Skills table
for index, row in df_courses_skills.iterrows():
    cur.execute("""
        INSERT INTO {0} (Link, Skill)
        VALUES (%s, %s);
    """.format(table_name), (row['Link'], row['Skills']))
    conn.commit()


# Bảng Categories
df_categories = merged_course[['MasterCategories','Categories']].copy()
df_categories  = df_categories .drop_duplicates()

# Define the Categories table
table_name = "Categories1"
cur.execute("""
    CREATE TABLE IF NOT EXISTS {0} (
        Category_id SERIAL PRIMARY KEY,
        Category VARCHAR(255) ,
        Parent_category VARCHAR(255)
    );
""".format(table_name))
conn.commit()

# Insert data into the Categories table
for index, row in df_categories.iterrows():
    cur.execute("""
        INSERT INTO {0} (Category, Parent_category)
        VALUES (%s, %s);
    """.format(table_name), (row['Categories'], row['MasterCategories']))
    conn.commit()


# Bảng Courses
df_courses = merged_course[['Link', 'Cluster', 'Categories', 'Descriptions', 'Level', 'Modules',
       'Name', 'NumEnrolled', 'NumReviews', 'Price', 'ReviewsURL', 'Skills',
       'Stars', 'Duration', 'InstructorName', 'ProviderName',
       'MasterCategories', 'Source']].copy()

# Define the Courses table
table_name = "courses1"
cur.execute("""
    CREATE TABLE IF NOT EXISTS {0} (
        Course_id SERIAL PRIMARY KEY,
        Link TEXT NOT NULL,
        Cluster VARCHAR(255),
        Categories VARCHAR(255),
        Descriptions TEXT,
        Level VARCHAR(255),
        Name VARCHAR(255),
        NumEnrolled VARCHAR(255),
        NumReviews VARCHAR(255),
        Price VARCHAR(255),
        ReviewsURL TEXT,
        Stars REAL,
        Duration VARCHAR(255),
        InstructorName VARCHAR(255),
        ProviderName VARCHAR(255),
        MasterCategories VARCHAR(255),
        Source VARCHAR(255)
    );
""".format(table_name))
conn.commit()

# Insert data into the Courses table
for index, row in df_courses.iterrows():
    cur.execute("""
        INSERT INTO {0} (Link, Cluster, Categories, Descriptions, Level,
        Name, NumEnrolled, NumReviews, Price, ReviewsURL,
        Stars, Duration, InstructorName, ProviderName,
        MasterCategories, Source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """.format(table_name), (row['Link'], row['Cluster'], row['Categories'], row['Descriptions'], row['Level'],
        row['Name'], row['NumEnrolled'], row['NumReviews'], row['Price'], row['ReviewsURL'],
        row['Stars'], row['Duration'], row['InstructorName'], row['ProviderName'],
        row['MasterCategories'], row['Source']))
    conn.commit()


# Bảng Instructors
df_instructors = merged_course["Instructor_full"].apply(eval).apply(pd.Series)
df_instructors = df_instructors.rename(columns={"instructor_name": "Instructor_name", "instructor_link": "Instructor_link"})
df_instructors["Bio"] = ""
df_instructors = df_instructors.drop_duplicates()

df_instructors= df_instructors.dropna(subset=["Instructor_link"])

# Define the Instructors table
table_name = "Instructors1"
cur.execute("""
    CREATE TABLE IF NOT EXISTS {0} (
        Instructor_id SERIAL PRIMARY KEY,
        Instructor_link TEXT NOT NULL,
        Instructor_name VARCHAR(255) ,
        Bio TEXT
    );
""".format(table_name))
conn.commit()

# Insert data into the Instructors table
for index, row in df_instructors.iterrows():
    cur.execute("""
        INSERT INTO {0} (Instructor_link, Instructor_name, Bio)
        VALUES (%s, %s, %s);
    """.format(table_name), (row['Instructor_link'], row['Instructor_name'], row['Bio']))
    conn.commit()


# Bảng Providers
df_providers = merged_course["Provider_full"].apply(eval).apply(pd.Series)
df_providers  = df_providers.rename(columns={"provider_name": "Provider_name", "provider_link": "Provider_link"})
df_providers ["Description"] = ""
df_providers  = df_providers .drop_duplicates()

df_providers= df_providers.dropna(subset=["Provider_link"])

# Define the Providers table
table_name = "Providers1"
cur.execute("""
    CREATE TABLE IF NOT EXISTS {0} (
        Provider_id SERIAL PRIMARY KEY,
        Provider_link TEXT NOT NULL,
        Provider_name VARCHAR(255) ,
        Description TEXT
    );
""".format(table_name))
conn.commit()

# Insert data into the Providers table
for index, row in df_providers.iterrows():
    cur.execute("""
        INSERT INTO {0} (Provider_link, Provider_name, Description)
        VALUES (%s, %s, %s);
    """.format(table_name), (row['Provider_link'], row['Provider_name'], row['Description']))
    conn.commit()



# Close the database connection
cur.close()
conn.close()