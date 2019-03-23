from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np
import time
import gensim

class NextSent:
    def __init__(self, pathList):
        self.pathList = pathList

    def __iter__(self):
        for file in self.pathList:
            with open(file.rstrip(), "r") as doc:
                for sent in doc.read().split("\n"):
                    yield sent.split()


def train_word2vec(vocabulary_inverse, final_paths, model_paths ="", model_save_path ="", option ="create"):
    model = None
    if option == "load":
        print('Loading Word2Vec model...')
        model = word2vec.Word2Vec.load(model_paths)
    else:
        with open(final_paths, "r") as paths:
            path_list = paths.read().strip().split()
            print("Paths reads", len(path_list))
            if option == "create":
                print('Creating Word2Vec model...')
                model = gensim.models.Word2Vec(iter=1, size=300, window=10, sg=1, sample=1e-5)
                model.build_vocab(NextSent(path_list))
                print("Vocabulary is builded!", time.time() - start, len(model.wv.vocab))
                model.train(NextSent(path_list), epochs=1, total_examples=model.corpus_count)
                model.save(model_save_path)
            elif option == "retrain":
                print('Retraining Word2Vec model...')
                model = word2vec.Word2Vec.load(model_paths)
                model.train(NextSent(path_list), epochs=1, total_examples=model.corpus_count)
                model.save(model_save_path)

    weights = {key: model[word] if word in model else np.random.uniform(-0.217, 0.217, model.vector_size)
               for key, word in vocabulary_inverse.items()}
    return weights