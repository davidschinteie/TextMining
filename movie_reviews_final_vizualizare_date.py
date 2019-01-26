# Partea 1: curatarea si prelucrarea datelor
reviews_train = []
for line in open('C:/Users/alexandru david/Desktop/folder-David/Master - IFR (baze de date si tehnologii WEB)/TextMining/movie_data/full_train.txt', 'r', encoding="utf8"):
    reviews_train.append(line.strip())
    
reviews_test = []
for line in open('C:/Users/alexandru david/Desktop/folder-David/Master - IFR (baze de date si tehnologii WEB)/TextMining/movie_data/full_test.txt', 'r', encoding="utf8"):
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
reviews_train_clean = preprocess_reviews(reviews_train)

# Etapa2:
# 2.a) eliminarea de stopwords
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

reviews_test_no_stop_words = remove_stop_words(reviews_test_clean)
reviews_train_no_stop_words = remove_stop_words(reviews_train_clean)

# 2.b) normalizarea cuvintelor (reducerea acestora la forma lor de baza)
# 2.b.1) Stemming
def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

reviews_test_stemmed_reviews = get_stemmed_text(reviews_test_no_stop_words)
reviews_train_stemmed_reviews = get_stemmed_text(reviews_train_no_stop_words)

# Lemmatization
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

reviews_test_lemmatized_reviews = get_lemmatized_text(reviews_test_stemmed_reviews)
reviews_train_lemmatized_reviews = get_lemmatized_text(reviews_train_stemmed_reviews)

reviews_test_final = reviews_test_lemmatized_reviews
reviews_train_final = reviews_train_lemmatized_reviews

# Partea a 2-a: Constructia clasificatorului:
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
# cv = CountVectorizer(binary=True, min_df=5, ngram_range=(1, 1))
# cv.fit(reviews_train_clean)
cv.fit(reviews_train_final)
# X = cv.transform(reviews_train_clean)
X = cv.transform(reviews_train_final)
# X_test = cv.transform(reviews_test_clean)
X_test = cv.transform(reviews_test_final)
print("Vocabulary size: {}".format(len(cv.vocabulary_)))
print("X_train:\n{}".format(repr(X)))
print("X_test: \n{}".format(repr(X_test)))

feature_names = cv.get_feature_names()
print("Number of features: {}".format(len(feature_names)))


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Crearea matricii de adevar pentru ceea ce se urmeaza a fi testat: o matrice cu primele .. valori setate cu 1 pentru review-uri pozitive si restul de .. valori setate cu 0 pentru review-uri negative
# target = [1 if i < 100 else 0 for i in range(200)]
target = [1 if i < 12500 else 0 for i in range(25000)]
# Scrierea matricii de adevar intr-un fisier:
# with open('C:/Users/alexandru david/Desktop/folder-David/Master - IFR (baze de date si tehnologii WEB)/TextMining/rezultate/matrice_de_adevar.txt', 'w', encoding="utf8") as f:
#     for item in target:
#         f.write("%s\n" % item)

param_grid = {'C': [0.001, 0.01, 0.05, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X, target)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))

# Partea a 4-a - Verificarea celor mai discriminatorii cuvintelor alese de algoritm in declararea review-urilor pozitive si celor negative:
import matplotlib.pyplot as plt
import mglearn
mglearn.tools.visualize_coefficients(grid.best_estimator_.coef_, feature_names, n_top_features=25)
plt.show()