## load embeddings in text, bin and given an input word, return it's nearest neighbours

from gensim import models
import sys
import codecs

def loadModel(vecFile, vecFormat):
	if vecFormat == "Text":
		model = models.KeyedVectors.load_word2vec_format(vecFile, binary=False, encoding='utf8')
	elif vecFormat == "Binary":
		model = models.KeyedVectors.load_word2vec_format(vecFile, binary=True, encoding='utf8')
	else:
		sys.exit ("Wrong format argument: enter either Text or Binary")
	
	return model


def find_similar_words(word):
	if word in word_vectors.vocab:
		return word_vectors.most_similar(word, topn=20)
	else:
		return None


def word_file_process(word_file, out_file):
	out = codecs.open(out_file, 'w', 'utf-8')
	with codecs.open(word_file, 'r', 'utf-8') as f:
		for word in f:
			sim_words = find_similar_words(word.rstrip())
			if sim_words == None: out.write(word.rstrip('\n') + "\t" + "WORD_NOT_IN_VOCAB" + "\n"); continue;
			output = word.rstrip('\n') + "\t"
			for i,j in sim_words:
                                output = output + i + " " + str(j) + "\t"
			out.write(output + "\n")
	f.close()
	out.close()
	return

def word_process():
	while True:
		word = raw_input("Enter a word to find its nearest neighbours: ").decode("utf-8")
                sim_words = find_similar_words(word)
                if sim_words == None: print "Word not in vocabulary"; continue;
                for i,j in sim_words:
                      print i,j
	return

if __name__ == "__main__":
	input_vector_file = sys.argv[1]
	input_vector_file_format = sys.argv[2] # binary or text
	
	# load model	
	model = loadModel(input_vector_file, input_vector_file_format)
	word_vectors = model.wv # if we only want query the model
	del model

	if len(sys.argv) > 3: # word file to get nearsest neighbors
		word_file = sys.argv[3]
		out_file = sys.argv[4]
		word_file_process(word_file, out_file)
	else:
		word_process()
		while True:
			word = raw_input("Enter a word to find its nearest neighbours:  Press 0 to EXIT ").decode("utf-8")
			if word == "0": break
			sim_words = find_similar_words(word)
			if sim_words == None: print "Word not in vocabulary"; continue;
			for i,j in sim_words:
				print i,j
				


