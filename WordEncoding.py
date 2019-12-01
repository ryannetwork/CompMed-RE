# encode sentense by BioWordVec in order to get tensor
from gensim.models import KeyedVectors
import json
import torch

class BioWordVec:
    def __init__(self, path = "/home/compmed/CompMed-RE/data/BioWordVec/bio_embedding_extrinsic"):
        print("Loading Word Embeddings from pretrained data ... ")
        self.path = path
        self.model = KeyedVectors.load_word2vec_format(self.path, binary=True)


# read data file
class RawData:
    def __init__(self, path="/home/compmed/CompMed-RE/processed_train.json"):
        print("Loading processed data ... ")
        self.path = path
        with open(path) as f:
            self.data = json.load(f)


class TextEncoder:
    def __init__(self, sentense, embeddings):
        self.embeddings = embeddings.model
        self.tokens = sentense.split()
        self.token_nums = len(self.tokens)
        self.embedding_dim = self.embeddings.wv.vectors.shape[1]

    def to_embeddings(self):
        # map self.sentence to a tensor
        vec_coll = []
        for word in self.tokens:
            try:
                vec = self.embeddings.get_vector(word).reshape((1, -1))
            except:
                vec = self.embeddings.get_vector("unk").reshape((1, -1))
            vec_coll.append(vec)
        vec_coll = np.concatenate(vec_coll)
        return vec_coll

all_instance = RawData()
word_model = BioWordVec()

print("Process into BioWordVec")
Big_tensor = []
for i in all_instnce.data:
    txt = TextEncoder(i['txt'], word_model)
    new_array = txt.to_embeddings().reshape(1, txt.token_nums, -1)
    Big_tensor.append(new_array)

Big_tensor = np.concatenate(Big_tensor)



