# Partea 1: curatarea si prelucrarea datelor
reviews_train = []
for line in open('C:/Users/alexandru david/Desktop/folder-David/Master - IFR (baze de date si tehnologii WEB)/TextMining/movie_data/small_train.txt', 'r', encoding="utf8"):
    reviews_train.append(line.strip())
    
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

# cv = CountVectorizer(binary=True)
cv = CountVectorizer(binary=True, min_df=5, ngram_range=(2, 2))
# cv.fit(reviews_train_clean)
cv.fit(reviews_train_final)
# X = cv.transform(reviews_train_clean)
X = cv.transform(reviews_train_final)
# X_test = cv.transform(reviews_test_clean)
X_test = cv.transform(reviews_test_final)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Crearea matricii de adevar pentru ceea ce se urmeaza a fi testat: o matrice cu primele .. valori setate cu 1 pentru review-uri pozitive si restul de .. valori setate cu 0 pentru review-uri negative
target = [1 if i < 100 else 0 for i in range(200)]
# target = [1 if i < 12500 else 0 for i in range(25000)]
# Scrierea matricii de adevar intr-un fisier:
# with open('C:/Users/alexandru david/Desktop/folder-David/Master - IFR (baze de date si tehnologii WEB)/TextMining/rezultate/matrice_de_adevar.txt', 'w', encoding="utf8") as f:
#     for item in target:
#         f.write("%s\n" % item)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

# Testarea acuratetii in functie de valorile diferite ale lui C:
# for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
#     lr = LogisticRegression(C=c)
#     lr.fit(X_train, y_train)
#     print ("Accuracy for C=%s: %s" 
#            % (c, accuracy_score(y_val, lr.predict(X_val))))
    
# Accuracy for C=0.01: 0.87104
# Accuracy for C=0.05: 0.87376
# Accuracy for C=0.25: 0.86784
# Accuracy for C=0.5: 0.86544
# Accuracy for C=1: 0.86336

# Partea a 3-a: Testarea algoritmului:
final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)

matrice_valori_teste = final_model.predict(X_test)
# Scrierea matricii cu valorile obtinute intr-un fisier:
# with open('C:/Users/alexandru david/Desktop/folder-David/Master - IFR (baze de date si tehnologii WEB)/TextMining/rezultate/matrice_valori_testate.txt', 'w', encoding="utf8") as f:
#     for item in matrice_valori_teste:
#         f.write("%s\n" % item)

print ("Final Accuracy: %s" 
        # obtinerea scorului prin compararea matricii de adevar cu matricea cu rezultatele obtinute in urma rularii testului
        % accuracy_score(target, final_model.predict(X_test)))


# Partea a 4-a - Verificarea celor mai discriminatorii cuvinte (seturi de cuvinte) alese de algoritm in declararea review-urilor pozitive si celor negative:
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}

cuvinte_pozitive = []
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:25]:
    cuvinte_pozitive.append(best_positive)
    
#     ('excellent', 0.9288812418118644)
#     ('perfect', 0.7934641227980576)
#     ('great', 0.675040909917553)
#     ('amazing', 0.6160398142631545)
#     ('superb', 0.6063967799425831)
    
cuvinte_negative = []
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:25]:
    cuvinte_negative.append(best_negative)
    
#     ('worst', -1.367978497228895)
#     ('waste', -1.1684451288279047)
#     ('awful', -1.0277001734353677)
#     ('poorly', -0.8748317895742782)
#     ('boring', -0.8587249740682945)


with open('C:/Users/alexandru david/Desktop/folder-David/Master - IFR (baze de date si tehnologii WEB)/TextMining/rezultate/cuvinte_populare_testate.txt', 'w', encoding="utf8") as f:
    f.write("----------------- Cuvinte pozitive: \n")
    for item in cuvinte_pozitive:
        f.write("{0}\n".format(item))
    f.write("----------------- Cuvinte negative: \n")
    for item in cuvinte_negative:
        f.write("{0}\n".format(item))

# import matplotlib.pyplot as plt
# import mglearn
# mglearn.tools.visualize_coefficients(grid.best_estimator_.coef_, feature_names, n_top_features=25)
# plt.show()