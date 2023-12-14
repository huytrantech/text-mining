from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer

def cosine():
    documents = [
        "I love natural language processing.",
        "Natural language processing is a subfield of artificial intelligence.",
        "Machine learning involves algorithms and statistical models."
    ]

    # Tạo ma trận TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Tính toán độ tương đồng cosine
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # In kết quả
    print("Cosine Similarity Matrix:")
    print(cosine_similarities)

def jaccard():
    documents = [
        "I love natural language processing.",
        "Natural language processing is a subfield of artificial intelligence.",
        "Machine learning involves algorithms and statistical models."
    ]

    # Tạo ma trận Bag of Words (BoW)
    count_vectorizer = CountVectorizer()
    bow_matrix = count_vectorizer.fit_transform(documents)

    array = bow_matrix.toarray()
    print(jaccard_score(array[0],array[1]))
jaccard()