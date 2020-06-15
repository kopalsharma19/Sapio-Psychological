# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:56:45 2020

@author: HITAKSHI SHAH
"""

import nltk
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
from pandas import DataFrame

URL = "https://www.newindianexpress.com/cities/hyderabad/2020/apr/11/lockdown-takes-a-heavy-toll-on-mentally-ill-2128584.html"
r = requests.get(URL).text

soup = BeautifulSoup(r,'lxml')
#print(soup.prettify())

headline = soup.find('h1').get_text()
# print(headline)

p_tags= soup.find_all('p')
p_tags_text = [tag.get_text().strip() for tag in p_tags]


sentence_list = [sentence for sentence in p_tags_text if not '\n' in sentence]
sentence_list = [sentence for sentence in sentence_list if '.' in sentence]
# Combine list items into string.
article = ' '.join(sentence_list)

# print(article)

URL2= "https://www.thehindu.com/news/cities/Hyderabad/calls-for-mental-health-counselling-on-the-rise/article31334357.ece"

r2= requests.get(URL2).text

soup2= BeautifulSoup(r2,'lxml')

headline2 = soup2.find('h1').get_text()

p_tags2= soup2.find_all('p')
p_tags_text2 = [tag.get_text().strip() for tag in p_tags2]


sentence_list2 = [sentence for sentence in p_tags_text2 if not '\n' in sentence]
sentence_list2 = [sentence for sentence in sentence_list2 if '.' in sentence]
# Combine list items into string.
article2 = ' '.join(sentence_list2)
#print(headline2)

URL3= "https://www.thenewsminute.com/article/hyderabad-mental-health-institute-sees-spike-alcoholics-withdrawal-symptoms-121502"
r3= requests.get(URL3).text

soup3= BeautifulSoup(r3,'lxml')

headline3 = soup3.find('h1').get_text()

p_tags3= soup3.find_all('p')

p_tags_text3 = [tag.get_text().strip() for tag in p_tags3]

li_tags3= soup3.find_all('li')
li_tags_text3 = [tag.get_text().strip() for tag in li_tags3]

p_tags_text3= p_tags_text3


sentence_list3 = [sentence for sentence in p_tags_text3 if not '\n' in sentence]
sentence_list3 = [sentence for sentence in sentence_list3 if '.' in sentence]
# Combine list items into string.
list_list3= ' '.join(li_tags_text3)

article3 = ' '.join(sentence_list3) + list_list3
#print(article3)



URL4= "https://www.thehindu.com/society/isolation-and-mental-health-the-psychological-impact-of-lockdown/article31237956.ece"
r4= requests.get(URL4).text

soup4= BeautifulSoup(r4,'lxml')

headline4 = soup4.find('h1').get_text()

p_tags4= soup4.find_all('p')
p_tags_text4 = [tag.get_text().strip() for tag in p_tags4]


sentence_list4 = [sentence for sentence in p_tags_text4 if not '\n' in sentence]
sentence_list4 = [sentence for sentence in sentence_list4 if '.' in sentence]
# Combine list items into string.
article4 = ' '.join(sentence_list4)
#print(article4)


URL5= "https://economictimes.indiatimes.com/industry/healthcare/biotech/healthcare/how-covid-19-has-affected-indias-mental-healthcare-support-system/articleshow/75098206.cms?from=mdr"
r5= requests.get(URL5).text

soup5= BeautifulSoup(r5,'lxml')

headline5 = soup5.find('h1').get_text()

p_tags5= soup4.find_all('p')
p_tags_text5 = [tag.get_text().strip() for tag in p_tags5]


sentence_list5 = [sentence for sentence in p_tags_text5 if not '\n' in sentence]
sentence_list5 = [sentence for sentence in sentence_list5 if '.' in sentence]
# Combine list items into string.
article5 = ' '.join(sentence_list5)
#print(article5)


URL6= "https://www.tribuneindia.com/news/comment/psycho-social-impact-of-the-covid-lockdown-69316"
r6= requests.get(URL6).text

soup6= BeautifulSoup(r6,'lxml')

headline6 = soup6.find('h1').get_text()

p_tags6= soup6.find_all('p')
p_tags_text6 = [tag.get_text().strip() for tag in p_tags6]


sentence_list6 = [sentence for sentence in p_tags_text6 if not '\n' in sentence]
sentence_list6 = [sentence for sentence in sentence_list6 if '.' in sentence]
# Combine list items into string.
article6 = ' '.join(sentence_list6)
#print(article6)

URL7= "https://www.moneycontrol.com/news/trends/health-trends/coronavirus-pandemic-mental-illness-cases-rise-in-india-after-covid-19-outbreak-study-5097231.html"
r7= requests.get(URL7).text

soup7= BeautifulSoup(r7,'lxml')

headline7 = soup7.find('h1').get_text()

p_tags7= soup7.find_all('p')
p_tags_text7 = [tag.get_text().strip() for tag in p_tags7]


sentence_list7 = [sentence for sentence in p_tags_text7 if not '\n' in sentence]
sentence_list7 = [sentence for sentence in sentence_list7 if '.' in sentence]
# Combine list items into string.
article7 = ' '.join(sentence_list7)
#print(article7)


URL8= "https://thefederal.com/the-eighth-column/with-covid-19-we-also-face-a-pandemic-of-anxiety/"
r8= requests.get(URL8).text

soup8= BeautifulSoup(r8,'lxml')

headline8 = soup8.find('h1').get_text()

p_tags8= soup8.find_all('p')
p_tags_text8 = [tag.get_text().strip() for tag in p_tags8]


sentence_list8 = [sentence for sentence in p_tags_text8 if not '\n' in sentence]
sentence_list8 = [sentence for sentence in sentence_list8 if '.' in sentence]
# Combine list items into string.
article8 = ' '.join(sentence_list8)
#print(article8)


URL9= "https://www.indiatoday.in/india/story/20-per-cent-increase-in-patients-with-mental-illness-since-coronavirus-outbreak-survey-1661584-2020-03-31"
r9= requests.get(URL9).text

soup9= BeautifulSoup(r9,'lxml')

headline9 = soup9.find('h1').get_text()

p_tags9= soup9.find_all('p')
p_tags_text9 = [tag.get_text().strip() for tag in p_tags9]


sentence_list9 = [sentence for sentence in p_tags_text9 if not '\n' in sentence]
sentence_list9 = [sentence for sentence in sentence_list9 if '.' in sentence]
# Combine list items into string.
article9 = ' '.join(sentence_list9)
#print(article9)

URL10= "https://www.msn.com/en-in/health/wellness/virus-lockdown-take-toll-on-mental-health-stress-anxiety-insomnia-rise-as-people-fret-over-future/ar-BB12aBJA"
r10= requests.get(URL10).text
soup10= BeautifulSoup(r10,'lxml')

headline10 = soup10.find('h1').get_text()

p_tags10= soup10.find_all('p')
p_tags_text10 = [tag.get_text().strip() for tag in p_tags10]


sentence_list10 = [sentence for sentence in p_tags_text10 if not '\n' in sentence]
sentence_list10 = [sentence for sentence in sentence_list10 if '.' in sentence]
# Combine list items into string.
article10 = ' '.join(sentence_list10)
#print(article10)



#word tokenizer
tokens = word_tokenize(article) + word_tokenize(headline) + word_tokenize(article2) + word_tokenize(headline2) + word_tokenize(article3) + word_tokenize(headline3) + word_tokenize(article4) + word_tokenize(headline4) + word_tokenize(article5) + word_tokenize(headline5) + word_tokenize(article6) + word_tokenize(headline6) + word_tokenize(article7) + word_tokenize(headline7) + word_tokenize(article8) + word_tokenize(headline8) + word_tokenize(article9) + word_tokenize(headline9)+ word_tokenize(article10) + word_tokenize(headline10)
# convert to lower case
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]

import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stop_words2= ['"','','us','et','al','also','faq','cdc','numbers', '.',
    ',',
    '--',
    '\'s',
    '?',
    ')',
    '(',
    ':',
    '\'',
    '\'re',
    '"',
    '-',
    '}',
    '{',
    u'—']
stripped = [w for w in stripped if not w in stop_words]
stripped = [w for w in stripped if not w in stop_words2]
# print(stripped)

print(Counter(stripped).most_common(500))

df= DataFrame(Counter(stripped).most_common(500),columns=['Word','Frequency'])
print(df)

df.to_csv('Words.csv')

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import pandas as pd


analyzer = SentimentIntensityAnalyzer()

sentiment = df['Word'].apply(analyzer.polarity_scores)
print(sentiment)

sentiment_df = pd.DataFrame(sentiment.tolist())
sentiment_df.head()
sentiment_df.tail()

def vaderize(df):
    '''Compute the Vader polarity scores for a textfield.
    Returns scores and original dataframe.'''

    analyzer = SentimentIntensityAnalyzer()

    print('Estimating polarity scores for %d cases.' % len(df))
    sentiment = df['Word'].apply(analyzer.polarity_scores)

    # convert to dataframe
    sdf = pd.DataFrame(sentiment.tolist()).add_prefix('vader_')
    # merge dataframes
    df_combined = pd.concat([df, sdf], axis=1)
    return df_combined

df_vaderized = vaderize(df)

df_vaderized['vader_compound'].plot(kind='hist')




import numpy as np 
import pandas as pd 


post_lockdown_words = pd.read_excel("C:/Users/HITAKSHI SHAH/Desktop/Post_lockdown  (Words).xlsx")

pd.isnull(post_lockdown_words).any()
post_lockdown_words['Score'].unique()
post_lockdown_words['Score'].value_counts()
post_lockdown_words.info()

post_lockdown_words['Word'] = post_lockdown_words.apply(lambda row: str(row['Word']).lower(), axis=1)
post_lockdown_words.head()
from string import punctuation
def remove_punctuations(string):
    return ''.join(c for c in string if c not in punctuation)

post_lockdown_words['Word'] = post_lockdown_words.apply(lambda row: remove_punctuations(row['Word']), axis=1)

def convert(integer):
    if(integer == 1):
        return 'Positive'
    else:
        return 'Negative'
    
    
post_lockdown_words['Score'] = post_lockdown_words.apply(lambda row: convert(row['Score']), axis=1)

from sklearn.model_selection import train_test_split

X = post_lockdown_words['Word']
y = post_lockdown_words['Score']

one_hot_encoded_label = pd.get_dummies(y)
one_hot_encoded_label.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(min_df=1, ngram_range=(1, 1))
X_train = vect.fit(X_train).transform(X_train) 
X_test = vect.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

c_val = [0.75, 1, 2, 3, 4, 5, 10]

for c in c_val:
    logreg = LogisticRegression(C=c)
    logreg.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_test, logreg.predict(X_test))))































