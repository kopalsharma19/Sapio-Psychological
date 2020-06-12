# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 20:07:38 2020

@author: HITAKSHI SHAH
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:56:45 2020

@author: HITAKSHI SHAH
"""

import requests
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
from pandas import DataFrame

URL = "https://www.telegraphindia.com/india/coronavirus-telangana-govt-launches-helpline-to-address-psychological-issues-during-lockdown/cid/1765973"
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

URL2= "https://telanganatoday.com/covid-19-pandemic-may-cause-long-term-health-problems-study"

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

URL3= "https://telanganatoday.com/covid-19-pandemic-angst-of-ophthalmologists"
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



URL4= "https://telanganatoday.com/61-indians-suffering-from-mental-health-issues-during-lockdown-survey"
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


URL5= "https://www.newindianexpress.com/cities/hyderabad/2020/apr/11/lockdown-takes-a-heavy-toll-on-mentally-ill-2128584.html"
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


URL6= "https://www.newindianexpress.com/cities/hyderabad/2020/may/30/hyderabad-3rd-in-mental-wellbeing-index-at-end-of-lockdown-30-2149776.html"
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

URL7= "https://www.thehindu.com/news/cities/Hyderabad/calls-for-mental-health-counselling-on-the-rise/article31334357.ece"
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


URL8= "https://www.thehindu.com/news/cities/Hyderabad/psychological-impact-of-pandemic-high-on-ophthalmologists-study/article31626676.ece"
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


URL9= "https://www.thenewsminute.com/article/hyderabad-mental-health-institute-sees-spike-alcoholics-withdrawal-symptoms-121502 "
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

URL10= "https://timesofindia.indiatimes.com/city/hyderabad/movement-restriction-may-have-adverse-impact-on-mental-and-social-well-being/articleshow/75782417.cms"
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


URL11= "https://science.thewire.in/health/pandemic-collective-anxiety-grieving-palliative-care/ "
r11= requests.get(URL11).text

soup11= BeautifulSoup(r11,'lxml')

headline11 = soup11.find('h1').get_text()

p_tags11= soup11.find_all('p')
p_tags_text11 = [tag.get_text().strip() for tag in p_tags11]


sentence_list11 = [sentence for sentence in p_tags_text11 if not '\n' in sentence]
sentence_list11 = [sentence for sentence in sentence_list11 if '.' in sentence]
# Combine list items into string.
article11 = ' '.join(sentence_list11)
#print(article11)


import nltk
nltk.download('punkt')


#wwwoord tokenizer
tokens = word_tokenize(article) + word_tokenize(headline) + word_tokenize(article2) + word_tokenize(headline2) + word_tokenize(article3) + word_tokenize(headline3) + word_tokenize(article4) + word_tokenize(headline4) + word_tokenize(article5) + word_tokenize(headline5) + word_tokenize(article6) + word_tokenize(headline6) + word_tokenize(article7) + word_tokenize(headline7) + word_tokenize(article8) + word_tokenize(headline8) + word_tokenize(article9) + word_tokenize(headline9)+ word_tokenize(article10) + word_tokenize(headline10) + word_tokenize(article11) + word_tokenize(headline11)
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

print(Counter(stripped).most_common(215))

df= DataFrame(Counter(stripped).most_common(215),columns=['Word','Frequency'])
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












