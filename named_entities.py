import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "Apple Inc. is planning to open a new store in San Francisco on January 1, 2023."

# Process the text with spaCy
doc = nlp(text)

# Print named entities
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")

# /ORG: Organization
# GPE: Geopolitical Entity