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

# afisarea cuvintelor obtinute intr-un fisier (pentru verificare):
with open('C:/Users/alexandru david/Desktop/folder-David/Master - IFR (baze de date si tehnologii WEB)/TextMining/rezultate/cleaned_small_test_v1.txt', 'w', encoding="utf8") as f:
    for item in reviews_test_clean:
        f.write("%s\n" % item)