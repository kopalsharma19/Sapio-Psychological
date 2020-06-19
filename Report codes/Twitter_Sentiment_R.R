data <- read.csv("hyderabad.csv", header= T ,stringsAsFactors = FALSE)
#Change dataset into a corpus
data_corp <- VCorpus(VectorSource(data))
#Data pre-processing
data_corp <- tm_map(data_corp, tolower)
data_corp <- tm_map(data_corp, PlainTextDocument)
data_corp <- tm_map(data_corp, removePunctuation)
for (i in seq(data_corp)) {
  data_corp[[i]] <- gsub('[^a-zA-Z|[:blank:]]', "", data_corp[[i]])
}
#Remove stop words
new_stops <-c("covid","iphone","coronavirus","hrefhttptwittercomdownloadandroid","relnofollowtwitter","androida","hrefhttptwittercomdownloadiphone","relnofollowtwitter","iphonea","web","rt","chuonlinenews","hrefhttpsmobiletwittercom","hrefhttptwittercomdownloadipad","bharianmy","lebih","berbanding","dijangkiti","kumpulan","mudah","terdedah","covidhttpstcoigdxdtmvrg","hrefhttpsabouttwittercomproductstweetdeck", "darah",'casesalaskaarizonaarkansascaliforniafloridakentuckyne'
              ,'uuu','can','des','appa','uuuu','false','true','positive')
#add here as per you data req



data_corp <- tm_map(data_corp, removeWords, words = c(stopwords("English"), new_stops))
data_corp <- tm_map(data_corp, stripWhitespace)
data_corp <- tm_map(data_corp, PlainTextDocument)
#Tokenize tweets texts into words


#library(rJava)  #IF REQUIRED
#Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre7')

#library(rJava)


install.packages("tokenizers")
library(tokenizers)


NLPtrigramTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)
}

tdm_NLP <- TermDocumentMatrix(data_corp, control=list(tokenize = NLPtrigramTokenizer))

data_cleaned_tdm_m <- as.matrix(tdm_NLP)
data_cleaned_freq <- rowSums(data_cleaned_tdm_m)

v <- sort(rowSums(data_cleaned_tdm_m),decreasing=TRUE)

d <- data.frame(word = names(v),freq=v)
head(d, 10)
#Word frequency analysis

install.packages('wordcloud')
install.packages ('RColorBrewer')
library(wordcloud)

library(RColorBrewer)
#Create a uni-gram (1-word) word cloud
pal <- brewer.pal(9,"Set1")
wordcloud(names(data_cleaned_freq), data_cleaned_freq, min.freq=50,max.words = 50, random.order=TRUE,random.color = TRUE, rot.per=.15, colors = pal,scale = c(3,1))
#change the 1s into the number of word-grams you would like to analyze
tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 1, max = 1))
}

install.packages("tidyr")
library(tidyr)
install.packages("janeaustenr")
install.packages("tidytext")
library(dplyr)

library(tidytext)
library(janeaustenr)

library(stopwords)
text
#SENTIMENT ANALYSIS
#Transform sentences into words
str(text)
data_tibble <- data %>%
  unnest_tokens(output = "Words", input = text, token = "words")
#Remove stop words from tibble
virus_tibble_clean <- data_tibble %>%
  anti_join(stop_words, by=c("words"="word"))

additional_sentiment <- tibble(word="positives",
                               sentiment="negative")
new_sentiment <- get_sentiments("bing")%>%
  rbind(additional_sentiment)
tail(new_sentiment)
dropwords <- "positives"
sentiments <- get_sentiments("bing") %>% filter(!word %in% dropwords)
install.packages("spread")
library(spread)
##Sentiment word frequency 
data_tidy_sentiment <- virus_tibble_clean %>%
  inner_join(sentiments, by = c("twwet" = "paragraphs")) %>%
  count(words, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(polarity = positive - negative)
summary(data_tidy_sentiment)
data_tidy_pol <- data_tidy_sentiment %>% 
  
  # Filter for absolute polarity at least 80 
  filter(abs(polarity) >= 1) %>% 
  # Add positive/negative status
  mutate(pos_or_neg = ifelse(polarity > 0, "positive", "negative"))
# Plot polarity vs. (term reordered by polarity), filled by pos_or_neg

install.packages("ggplot2")
library(ggplot2)

ggplot(data_tidy_pol, aes(reorder(words, polarity), polarity, fill = pos_or_neg)) +geom_col() + 
  ggtitle("Coronavirus related tweets: Sentiment Word Frequency") + 
  # Rotate text and vertically justify
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, size = 10))+
  xlab("Word")

#Most positive and negative words used in the extracted tweets



word_counts <- virus_tibble_clean %>%
  # Implement sentiment analysis using the "bing" lexicon
  inner_join(sentiments, by = c("words" = "word")) %>%
  # Count by word and sentiment
  count(words, sentiment)
top_words <- word_counts %>%
  # Group by sentiment
  group_by(sentiment) %>%
  # Take the top 10 for each sentiment
  top_n(10) %>%
  ungroup() %>%
  # Make word a factor in order of n
  mutate(words = reorder(words, n))
# Use aes() to put words on the x-axis and n on the y-axis
ggplot(top_words, aes(words, n, fill = sentiment)) +
  # Make a bar chart with geom_col()
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = n, hjust=1), size = 3.5,  color = "black") +
  facet_wrap(~sentiment, scales = "fixed",as.table = F) +  
  coord_flip() +
  ggtitle("Most common positive and negative words")



#Sentiment word cloud
install.packages("textdata")
library(textdata)



data_tidy <- virus_tibble_clean %>%
  # Inner join to nrc lexicon
  inner_join(get_sentiments("nrc"), by = c("words" = "word")) %>% 
  # Drop positive or negative
  filter(!grepl("positive|negative", sentiment)) %>% 
  # Count by sentiment and term
  count(sentiment, words) %>% 
  # Spread sentiment, using n for values
  spread(sentiment, n, fill = 0)  %>% 
  # Convert to data.frame, making term the row names
  data.frame(row.names = "words")
# Plot comparison cloud
comparison.cloud(data_tidy, max.words = 130, title.size = 1)


