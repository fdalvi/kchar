## Input embeddings in text or binary format and find the closest words

### Requirements
Code is written in python 2.7

For Text format, here embedding is the file that has embeddings in word2vecf format. One word per line with embeddings separated by space. First line of the file has total number of words and embedding size
```
python find_similar_words.py embedding Text
```

For binary word2vec file:
```
python find_similar_words.py embedding Binary
```

A word list can be provided to get the similar words for all words in the list
```
python find_similar_words.py embedding Text input_file output_file
```


