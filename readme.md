- readme legat de date: 
    - revie-urile au fost descarcate de pe <a href="http://ai.stanford.edu/~amaas/data/sentiment/">Stanford: Large Movie Review Dataset</a> contine un set de training (25k - 12.5k pozitive si 12.5k negative) si un set de testare (25k - 12.5k pozitive si 12.5k negative).
    ---
    Partea 1: curatarea si prelucrarea datelor
    full_test.txt: 12500 review-uri pozitive si 12500 review-uri negative pentru testare
    full_train.txt: 12500 review-uri pozitive si 12500 review-uri negative pentru training/invatare 
    Accuracy: 0.88152 - capture01_b.png
    ---
    Etapa1: review-urile din full_test curatate de taguri HTML, de semne de punctatie, au fost reduse la lower-case (etapa 1) cleaned_small_test_v1.txt (fisier cu primele 200 de revie-uri prelucrate)
    Astfel ca urmatorul comentariu: "I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. This one did exactly that. The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half. While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying. This movie was great, and I suggest that you go see it before you judge." a fost prelucrat in urmatoarea forma: "i went and saw this movie last night after being coaxed to by a few friends of mine ill admit that i was reluctant to see it because from what i knew of ashton kutcher he was only able to do comedy i was wrong kutcher played the character of jake fischer very well and kevin costner played ben randall with such professionalism the sign of a good movie is that it can toy with our emotions this one did exactly that the entire theater which was sold out was overcome by laughter during the first half of the movie and were moved to tears during the second half while exiting the theater i not only saw many women in tears but many full grown men as well trying desperately not to let anyone see them crying this movie was great and i suggest that you go see it before you judge"
    ----
    In etapa 2 au fost eliminate stopwords si cuvintele au fost normalizate (eliminarea terminatiilor diferite pentru acelasi cuvant de baza)
    Astfel ca primul comentariu a fost prelucrat in urmatoarea forma: "went saw movi last night coax friend mine ill admit reluct see knew ashton kutcher abl comedi wrong kutcher play charact jake fischer well kevin costner play ben randal profession sign good movi toy emot one exactli entir theater sold overcom laughter first half movi move tear second half exit theater saw mani woman tear mani full grown men well tri desper let anyon see cri movi great suggest go see judg"
    Accuracy: 0.87676 - capture02_b.png

- readme legat de algoritm (Logistic Regression):
    1) Vectorizarea datelor
    Pentru a putea transmite datele algoritmului de machine learning va trebui sa asociem fiecarui review un numar pentru o reprezentare vectoriala a datelor.
    
    Metoda aleasa a fost de a crea o matrice cu o coloana pentru fiecare cuvant unic din cele 50k de revie-uri. Apoi convertim fiecare review in parte intr-un singur rand cu 0 si 1 - unde 1 inseamna ca respectivul cuvant din coloana matricii apare in review. Asta inseamna ca fiecare rand din matrice va fi majoritar completata cu valoarea 0. Proces cunoscut si sub numele de one hot encoding. 

    2) Alegerea parametrului C (1/lambda):
    Pentru valori mici ale lui C creste valoarea lui lambda care va face ca modelul sa fie foarte simplu (underfit data) -- daca in schimb C are valori foarte mari atunci lambda va lua valori foarte mici ceea ce va creste complexitatea modelului (overfit data).
    Se va testa setul de date cu diferite valori ale lui C (0.01, 0.05, 0.25, 0.5, 1) si se alege valoarea corespunzatoare celui mai mare scor al acuratetii.
    In cazul acestui set de date rezultatele obtinute pentru C au fost:
    # Accuracy for C=0.01: 0.87104
    # Accuracy for C=0.05: 0.87376
    # Accuracy for C=0.25: 0.86784
    # Accuracy for C=0.5: 0.86544
    # Accuracy for C=1: 0.86336
    Rezulta ca C=0.05 va fi ales ca valoare -> capture02_b.png

    3) Alegerea parametrilor pentru Vectorizare:
    min_df ( = 5): parametru care stabileste frecventa minima a unui cuvant inainte a fi setat ca si feature
    ngram_range=(1, 2): parametru care defineste lungimea maxima si minima a unui set de cuvinete care pot fi setate ca si features
    
    CountVectorizer(binary=True)
    Final Accuracy: 0.87676 - capture02_b.png
    
    CountVectorizer(binary=True, min_df=5, ngram_range=(1, 2))
    Final Accuracy: 0.88736 - capture03_b.png

    CountVectorizer(binary=True, min_df=5, ngram_range=(2, 2))
    Final Accuracy: 0.84848 - capture03_c.png

    4)Verificarea celor mai discriminatorii cuvinte (seturi de cuvinte) alese de algoritm in declararea review-urilor pozitive si celor negative (primele 25 de feature-uri):
    -> Vizualizarea datelor (capture04_a.png, capture04_b.png si capture04_c.png)