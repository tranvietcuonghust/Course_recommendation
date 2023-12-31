import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities import MatrixSimilarity

merged_course = pd.read_csv('./data/merged_course_final.csv')

merged_course['Keywords'] = merged_course['Keywords'].astype(str)

# Processing Keywords
keywords = merged_course['Keywords'].tolist()

keywords = [word_tokenize(keyword.lower()) for keyword in keywords]

def no_commas(doc):
    no_commas = [t for t in doc if t!=',']
    return(no_commas)

keywords = [no_commas(kw) for kw in keywords]
processed_keywords = keywords

dictionary = Dictionary(processed_keywords) # create a dictionary of words from our keywords

corpus = [dictionary.doc2bow(doc) for doc in processed_keywords]
#create corpus where the corpus is a bag of words for each document

tfidf = TfidfModel(corpus) #create tfidf model of the corpus

# Create the similarity data structure. This is the most important part where we get the similarities between the movies.
sims = MatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

def keywords_recommendation(keywords, number_of_hits):
    query_doc_bow = dictionary.doc2bow(keywords) # get a bag of words from the query_doc
    query_doc_tfidf = tfidf[query_doc_bow] #convert the regular bag of words model to a tf-idf model where we have tuples
    # of the movie ID and it's tf-idf value for the movie

    similarity_array = sims[query_doc_tfidf] # get the array of similarity values between our movie and every other movie.
    #So the length is the number of movies we have. To do this, we pass our list of tf-idf tuples to sims.

    similarity_series = pd.Series(similarity_array.tolist(), index=merged_course.Link.values) #Convert to a Series
    top_hits = similarity_series.sort_values(ascending=False)[:number_of_hits] #get the top matching results,
    # i.e. most similar movies

    # # Print the top matching movies
    # print("Our top %s most similar course for the keywords %s are:" %(number_of_hits, keywords))
    # for idx, (movie,score) in enumerate(zip(top_hits.index, top_hits)):
    #     print("%d '%s' with a similarity score of %.3f" %(idx+1, movie, score))
    Names = merged_course.set_index('Link')['Name']

    # Trả về danh sách các khóa học được đề xuất
    result = [{"course_link": course,"course_name" :Names[course], "similarity_score": score} for course, score in zip(top_hits.index, top_hits)]
    return result

# keywords_recommendation(['javascript','frontend','web','database'], 10)