import pandas as pd
import re
import csv
from nltk.corpus import stopwords
from pickle import dump, load
import nltk

reviews = pd.read_csv('data/hackernoon_tutorial/Reviews.csv')
print(reviews.shape)
print(reviews.head())
print(reviews.isnull().sum())

reviews = reviews.dropna()
reviews = reviews.drop(['Id',
                        'ProductId',
                        'UserId',
                        'ProfileName',
                        'HelpfulnessNumerator',
                        'HelpfulnessDenominator',
                        'Score',
                        'Time']
                       , 1)
reviews = reviews.reset_index(drop=True)
print(reviews.head())
for i in range(5):
    print('Review #', i + 1)
    print(reviews.Summary[i])
    print(reviews.Text[i])
    print()

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


def clean_text(text, remove_stopwords=True):
    # Convert words to lower case
    text = text.lower()
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'<br >', ' ', text)
    text = re.sub(r'\'', ' ', text)

    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words('english'))
        text = [w for w in text if not w in stops]
        text = ' '.join(text)

    return text


nltk.download('stopwords')

clean_summaries = []
c = 0
for summary in reviews.Summary:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
    c = c + 1
    if c % 1000 == 0:
        print('Processing summaries [', c, ']')
    if c % 2000 == 0:
        break
print('Summaries are complete.')

clean_texts = []
c = 0
for text in reviews.Text:
    clean_texts.append(clean_text(text))
    c = c + 1
    if c % 1000 == 0:
        print('Processing texts [', c, ']')
    if c % 2000 == 0:
        break
print('Texts are complete.')

stories = list()
stories2 = [['Story', 'Highlights']]
for i, text in enumerate(clean_texts):
    stories.append({'story': text, 'highlights': clean_summaries[i]})
    stories2.append([text, clean_summaries[i]])

with open('C:/Users/Konrad/PycharmProjects/text_summarization/data/hackernoon_tutorial/review_dataset.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile, delimiter=',', quoting=csv.QUOTE_ALL)
    writer.writerows(stories2)

writeFile.close()

dumpfile = open('C:/Users/Konrad/PycharmProjects/text_summarization/data/hackernoon_tutorial/review_dataset.pkl', 'wb')
dump(stories, dumpfile)
dumpfile.close()
