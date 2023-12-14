import spacy
from nltk.stem import PorterStemmer,SnowballStemmer

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize the NLTK Porter Stemmer
# stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
# Example text
text = "running and eating,ate are two common activities"

# Process the text with spaCy
doc = nlp(text)

# Perform stemming using NLTK
stemmed_words = [stemmer.stem(token.text) for token in doc]

# Print the result
print("Original text:", text)
print("Stemmed text:", " ".join(stemmed_words))


sentence6 = nlp(u'running and eating,ate are two common activities')
lemmed_words = [word.lemma_ for word in sentence6]
# Print the result
print("Original text:", text)
print("Lemmed text:", " ".join(lemmed_words))