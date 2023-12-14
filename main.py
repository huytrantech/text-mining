import math

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# Dữ liệu mẫu
documents = [
    "watch King’ Speech",
    "King’ Speech",
    "decid watch movi"
]

# Tạo một đối tượng TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(use_idf=True,norm=None,smooth_idf=True)

# Biến đổi dữ liệu văn bản thành ma trận TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
# Xuất thông tin từ điển của TfidfVectorizer
print("Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())


# get idf values
print('\nidf values:')
for ele1, ele2 in zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_):
	print(ele1, ':', ele2)


# Xuất ma trận TF-IDF
print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

# If ``smooth_idf=True`` (the default), the constant "1" is added to the
#     numerator and denominator of the idf as if an extra document was seen
#     containing every term in the collection exactly once, which prevents
#     zero divisions: idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.

print(1 * (math.log(4/3) + 1))
print(1 * (math.log(4/2) + 1))
print(1 * math.log(3/2))