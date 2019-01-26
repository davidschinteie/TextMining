# Partea 1: curatarea si prelucrarea datelor
reviews_test = []
for line in open('C:/Users/alexandru david/Desktop/folder-David/Master - IFR (baze de date si tehnologii WEB)/TextMining/movie_data/small_test.txt', 'r', encoding="utf8"):
    reviews_test.append(line.strip())

import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

# Etapa 1:
# curatarea review-urilor prin eliminarea de caractere de punctuatie, caractere HTML si prin transformarea cuvintelor in lower-case
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

reviews_test_clean = preprocess_reviews(reviews_test)

# Etapa2:
# 2.a) eliminarea de stopwords
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')
print (english_stop_words)

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

no_stop_words = remove_stop_words(reviews_test_clean)

# 2.b) normalizarea cuvintelor (reducerea acestora la forma lor de baza)
# 2.b.1) Stemming
def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

stemmed_reviews = get_stemmed_text(no_stop_words)

# Lemmatization
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

lemmatized_reviews = get_lemmatized_text(stemmed_reviews)

# afisarea matricii obtinute intr-un fisier (pentru verificare):
with open('C:/Users/alexandru david/Desktop/folder-David/Master - IFR (baze de date si tehnologii WEB)/TextMining/rezultate/cleaned_small_test_v2.txt', 'w', encoding="utf8") as f:
    for item in lemmatized_reviews:
        f.write("%s\n" % item)