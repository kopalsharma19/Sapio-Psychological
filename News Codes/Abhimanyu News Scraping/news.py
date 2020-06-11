import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from pandas import DataFrame

URL = "https://www.news-medical.net/news/20200609/COVID-19-survivors-could-suffer-long-term-health-effects.aspx"
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

URL2= "https://nenow.in/health/how-covid-19-impacts-the-psycho-social-health-of-the-most-vulnerable.html"
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

URL3= "https://www.cdc.gov/coronavirus/2019-ncov/daily-life-coping/managing-stress-anxiety.html"
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



URL4= "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7151415/"
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



tokens = word_tokenize(article) + word_tokenize(headline) + word_tokenize(article2) + word_tokenize(headline2) + word_tokenize(article3) + word_tokenize(headline3) + word_tokenize(article4) + word_tokenize(headline4)
# convert to lower case
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]


stop_words = set(stopwords.words('english'))
stop_words2= ['','us','et','al','also','faq','cdc']
stripped = [w for w in stripped if not w in stop_words]
stripped = [w for w in stripped if not w in stop_words2]
# print(stripped)

print(Counter(stripped).most_common(100))

df= DataFrame(Counter(stripped).most_common(100),columns=['Word','Frequency'])
print(df)

df.to_csv('Words.csv')


