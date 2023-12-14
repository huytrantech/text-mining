import spacy

# Tải mô hình ngôn ngữ tiếng Anh của spaCy
nlp = spacy.load("en_core_web_sm")

# Ví dụ câu
sentence = "Mary can see Adam"

# Xử lý câu với spaCy
doc = nlp(sentence)

# Trích xuất nhãn Parts of Speech

pos_tags = [(token.text, token.pos_) for token in doc]

print(pos_tags)
