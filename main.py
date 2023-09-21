import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

#DataFrames
movies = pd.read_csv('./tmdb_5000_movies.csv')
credit = pd.read_csv('./tmdb_5000_credits.csv')

#New Data Frame
movies = movies.merge(credit,on='title')

#Final Raw Data Frame
movies = movies[['id','title','genres','keywords','overview','cast','crew']]
#Jahan Jahan data empty hoga us ko drop kr denge
movies.isnull().sum()
movies.dropna(inplace=True)
#checking Duplicate Data (No Duplicate Data Found)
movies.duplicated().sum()

#Data Converting into List from Dictonary
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

#Function for top 3 celebrities of a movie (Dictonary -> List)
def convert3(obj):
    L = []
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

#fetching only Director's name from crew Column
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

#converting overview's string into List
movies['overview'] = movies['overview'].apply(lambda x:x.split())

#Removing space from a single entity (Rishi Jain = RishiJain) for almost perfect results of model
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

#Concatenating
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

#New Data Frame
new_df = movies[['id','title','tags']]

#List -> String (Not Working) (Error de rah)
#new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
#new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
#print(new_df.head())

def ltos(obj):
      str1 = " "
      return (str1.join(obj))

#List -> String -> Small Letters
new_df['tags'] = new_df['tags'].apply(ltos)
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

ps = PorterStemmer()

#(danced,dance,dancing =dance) pehle list banegi fir String me convert karke return
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

#MaxFeatures = Maximum Occuring words , StopWords = (is,are,for,from etc) [cv ek object hai]
cv = CountVectorizer(max_features=5000,stop_words='english')
#cv use karne pr -> Sparse Matrix -> Numpy Array (Vector contains spase Matrix now)
vectors = cv.fit_transform(new_df['tags']).toarray()

#Calculate similarity of movie x with every movie(4806)
similartiy = cosine_similarity(vectors)


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distance = similartiy[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

print(recommend('Iron Man 3'))

file = "movies.pkl"
fileobj = open(file,'wb')
pickle.dump(new_df,fileobj)
#pickle.dump(new_df,open('movies.pkl','wb'))

file2 = "similarity.pkl"
fileobj2 = open(file2,'wb')
pickle.dump(similartiy,fileobj2)

