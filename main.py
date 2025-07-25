# music recommmender system
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

#EDA Exploratory data analysis (get a feel of the data)
data = pd.read_csv('tcc_ceds_music.csv')
track_names = data['track_name']
# # 
# print(track_names.iloc[0]) # outputs the track name given index, baiscally data['track_name'][0]

print(data.head()) # first 5 rows of stuff

plt.figure(figsize=(10,6))
sns.countplot(y='genre', data=data, order=data['genre'].value_counts().index[:10])
plt.title('Top 10 genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()


data['combined_features'] = (
   data['genre'].fillna('') + ' ' +
   data['artist_name'].fillna('') + ' ' +
   data['track_name'].fillna('')
) # basically concatnates all tat information into 1 string and .fillna prevents errors from empty columns
# example: Genre: Pop, artist_name: Taylor Swist, track_name: Shake it Off
# data['combined_features'] = 'Pop Taylor Swift Shake It Off'
# also is vecotirzed operations where it does everything at once (basically iterable but it doesn't iterate and does everything at once)
# Result: data['combined_features'][0] == ['Pop Taylor Swift Shake It Off']
#         data['combined_features'][1] == ["Rock Foo Fighters The Rock Anthem"]
#         data['combined_features'][2] == ["Pop Dua Lipa Love Again"]
tfidf = TfidfVectorizer(stop_words='english')
# remove words such as it, off, the
# Result: data['combined_features'][0] == ['Pop Taylor Swift Shake']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
#         data['combined_features'][1] == ["Rock Foo Fighters Rock Anthem"]
#         data['combined_features'][2] == ["Pop Dua Lipa Love Again"]


# basically removes the common words which don't provide any useful info 


tfidf_matrix = tfidf.fit_transform(data['combined_features'])
# IDF scores are given to each word
# a high idf score means it is unique and doesn't appear frequently or only once in everything
# Ex: pop has a low IDF because it appears twice, while every other word has a high IDK because it appears only once                                  


# TF: term frequency
# how often a term t occurs in a document d
# pop in song 1 has tf of 1 and pop in song 3 has a tf of 1
# rock in song 2 has a tf of 2
# Everything else has TF of 1
# everything is formatted in a matrix with words as column headers and song number has row headers

# TF matrix
#            pop Taylor Swift Shake Rock Foo Fighters Anthem Dua Lipa Love Again
# song 1     1     1      1     1    0    0     0        0    0   0     0    0
# song 2     0     0      0     0    2    1     1        1    0   0     0    0
# song 3     1     0      0     0    0    0     0        0    1   1     1    1

# IDF Inverse Document frequency
#Rarity score
# higher IDF =  rarer
# Lower IDDDDF = more common
# another matrix with same headers is constructued of IDF values
# calculated through log ( 1 + total documents / 1 + total documents containing t) + 1
# for esxample pop: log ( 1 + 3 / 1 + 2) + 1 = log(4/3) + 1 = 1.287682
# IDF for taylor: log (1 + 3 / 1 + 1) + 1 = log(2) + 1 = 1.693147
# IDF for rock: log(1+3 / 1 + 1) + 1 = 1.69... its + 1 in denominator because rock is only found in 1 document, song b

#IDF (not a matrix, because all metrics are global and not by document)
#     pop   Taylor Swift Shake Rock Foo  Fighters Anthem Dua  Lipa Love Again
#     1.28  1.69   1.69  1.69  1.69 1.69 1.69     1.69   1.69 1.69 1.69 1.69


#TFIDF score = TF * IDF and then goes through l2 normalization
# high TFIDF score is desireable
# for example out of 100K random news article, the word quantum might have a high TFIDF
# this is because it appears is few articles, but is mentioned many times during those few articles
# Now a raw TF-IDF matrix is constructed
# multiply the 2 matrixes tgt
# column pop row song 1 = 1 * 1.287 (TF * IDF)
# column rock row song 2 =  2 * 1.693 = 3.386 (TF * IDF)                           
# column dua row song 3 = 1 * 1.693 = 3.386 (TF * IDF)  

#            pop   Taylor Swift Shake Rock Foo  Fighters Anthem Dua  Lipa Love Again
# song 1     1.28  1.69   1.69  1.69  0    0    0        0      0    0    0    0
# song 2     0     0      0     0     3.38 1.69 1.69     1.69   0    0    0    0
# song 3     1.28   0      0     0    0    0     0        0     1.69 1.69 1.69 1.69


# last step, each row (document) is divided by its L2-norm
# the reason is longer document (song 1, song 2, etc.) will have higher TF due to it having a higher chance of repeating words
# this step scales all documents to same length, removes influence of (potentially biased) document length
# levels the playing field
# if this process is repeated 1 more time, it should yield 1
# forumla: sqrt( v1 ^2 + v2^2 + ...)
# v1 is the TFIDF score of pop, v2 is shake, swift, taylor (which v is which word doens't matter)
# song 1 L2 norm: sqrt ( [1.287]^2 + [1.693]^2 + [1.693]^2 + [1.693]^2 ) = 3.20289
# song 2 l2 norm: sqrt([1.693] ^2 + [1.693] ^2 + [1.693] ^2 + [3.386] ^2 ) = 4.366
# song 3 l2 norm: sqrt ( [1.287]^2 + [1.693]^2 + [1.693]^2 + [1.693]^2 + [1.693]^2 )= 3.61424958
# etc. etc, song 3 l2 norm: 3.622
# now divide each value in row of IDF by the respective l2 norms
# ex: pop in row 1: 1.287/3.2 (TFIDF / L2) = 0.402
# full TDIDF of song 1 = [0.402, 0.5286, 0.5286, 0.5286]
# if we l2-norm it again, it will all equal 1
# this is same for all, thus the length of the vector is irrelavent
# now only the direction of the vetor matters

#            pop   Taylor Swift Shake Rock Foo  Fighters Anthem Dua  Lipa Love Again
# song 1     1.28  1.69   1.69  1.69  0    0    0        0      0    0    0    0     <-- everything in this row is divided by 3.20
# song 2     0     0      0     0     3.38 1.69 1.69     1.69   0    0    0    0     <-- everything in this row is divided by 4.36
# song 3     1.28   0      0     0    0    0     0        0     1.69 1.69 1.69 1.69  <-- everything in this row is divided by 3.61

# l2 normalized matrix
#            pop   Taylor Swift Shake Rock Foo  Fighters Anthem Dua  Lipa Love Again
# song 1     0.4   0.528  0.528 0.528 0    0    0        0      0    0    0    0    
# song 2     0     0      0     0     0.75 0.37 0.37     0.37   0    0    0    0    
# song 3     0.35  0      0     0     0    0     0        0     0.47 0.47 0.47 0.47 

#L2-norm represnts length/magnitude of the vector
# 2d vector of [3,4] has vector of 5
# it also improves cosine similarty score


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# print(cosine_sim[0][0])
# dot product is performed
# basically meaning each index of each TFIDF vector is multipled with every other vector
# Ex: the TFIDF value for anthem from song 1 is mulitplied from anthem 2 ( 0 x 0 = 0)
# Ex: the TFIDF value for fighters from song 1 * from song 2 = (0 * 0.377) = 0
# Ex: the TFIDF value for pop from song 1 * song 3 = 0.402 * 0.402 = 0.161
# basically using dot vector, everything is added up to find relationships between every document (rows) with each other

# Ex: song 1 and song 2
# song 1     0.4   0.528  0.528 0.528 0    0    0        0      0    0    0    0    
# song 2  x  0     0      0     0     0.75 0.37 0.37     0.37   0    0    0    0   
# -------------------------------------------------------------------------------- 
#            0  +  0  +   0  +  0  +  0 +  0  + 0   +    0  +   0 +  0 +  0  + 0 = 0  

# Ex: song 1 and song 3
# song 1     0.4   0.528  0.528 0.528 0    0    0        0      0    0    0    0    
# song 3  x  0.35  0      0     0     0    0     0        0     0.47 0.47 0.47 0.47  
# -------------------------------------------------------------------------------- 
#            0.16  +  0  +   0  +  0  +  0 +  0  + 0   +    0  +   0 +  0 +  0  + 0 = 0.16  

# the final consine_sim matrix:
#        song 1 song 2 song 3
# song 1   1      0     0.16
# song 2   0      1       0
# song 3  0.16    0       1
# the 0.16 is the result of the 2 pops
# 1 is a high score, meaning high correlation, 0 is no correlation




# for i in range(len(cosine_sim)):
#    highest = 0
#    name = data['track_name'][0]
#    for j in range(i):
#       if cosine_sim[i][j] == 1:
#          pass
#       elif cosine_sim[i][j] > highest:
#          highest = cosine_sim[i][j]
#          name = 
#    a = cosine_sim[i]
#    cop = a.copy()
#    cop[i] = -1
#    most_same_idx = np.argmax()
#    print(track_names[most_same_idx + 1])


desired_track_name = 'cry'
matching_indice = track_names[track_names == desired_track_name].index[0] # there are many so we need to identify which one tbh, for now just the first one

highest = 0
name = data['track_name'][0]
author = data['artist_name'][0]
for j,i in enumerate(cosine_sim[matching_indice]):
    # print(f"j:{j}") # works fine, prints out the index
    # print(f"i:{i}") # works fine prints out the l2 normalized tfidf value
    if (i < 1) and i > highest:
        highest = i
        name = data['track_name'][j]
        author = data['artist_name'][j]
        print(f"name: {name}")
        print(f"cosine similiarty: {i}")

print(f"The recommended song is: {name.title()} by {author.title()}")
#Iterate through cos
