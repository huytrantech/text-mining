import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")
nlp.Defaults.stop_words.add("example")
# Example text
text = "This is an example sentence with some stop words."

# Process the text with spaCy
doc = nlp(text)

# Remove stop words and join the remaining tokens
filtered_tokens = [token.text for token in doc if not token.is_stop]
filtered_text = ' '.join(filtered_tokens)

# Print the result
print("Original text:", text)
print("Text after stop word removal:", filtered_text)

stop_words = nlp.Defaults.stop_words

# Print the list of stop words
print("List of stop words in English:", stop_words)