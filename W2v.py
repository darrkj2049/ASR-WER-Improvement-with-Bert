from gensim.models import word2vec

#gerneral settings
seed = 1200 # random seed numnber
sg = 0 #selection of CBOW=0/Skip-gram=1
window = 10 #the distant between current and predicted word within a sentence
vector_size = 100 #dimension of the word vectors (how many layers of a word)
min_count = 1 #the min words(quantity) wont act as vector
workers = 8 #parallel layers number
epochs = 5 #forwards number
batch_words = 10000 #how many training for each words

train_data = word2vec.LineSentence('wiki_text.txt')
model = word2vec.Word2Vec(
    train_data,
    min_count = min_count,
    vector_size = vector_size,
    workers=workers,
    epochs=epochs,
    window=window,
    sg=sg,
    seed=seed,
    batch_words=batch_words    
)

model.save('test_w2v.model')