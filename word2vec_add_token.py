
# coding: utf-8

# 將PTT的語料進行斷詞

# In[5]:


dir_path = r"D:\Backup\ml_data\GitHub\chat-robot_seq2seq\word2vec_model"
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Jieba_file_segmentation, replace_old=True,file_filter_='ptt_data')


# 句子前後都已經加上 < EOS >、 < SOS > 的token，將用wiki訓練的model使用這次要訓練的資料，訓練新的字詞。

# In[28]:


from gensim.models import word2vec
from ML_utils import ML_utils
import logging
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
word2vec_modle_path = r"D:\Backup\ml_data\GitHub\chat-robot_seq2seq\word2vec_model\word2vec_model"
model = word2vec.Word2Vec.load(word2vec_modle_path)
n_word = print(len(model.wv.index2word))
try:
    model.wv.most_similar(positive='<SOS>',topn=1)
except KeyError as err:
    print(err)
try:
    model.wv.most_similar(positive='<EOS>',topn=1)
except KeyError as err:
    print(err)    
model.vocabulary.min_count = 5
file_path = r"D:\Backup\ml_data\GitHub\chat-robot_seq2seq\word2vec_model\ptt_train_vec.txt"
sentences = word2vec.LineSentence(file_path)
model.build_vocab(sentences, update=True)#训练该行
model.train(sentences,total_examples= model.corpus_count,epochs= model.epochs)
print(len(model.wv.index2word))
print(model.wv.most_similar(positive='<SOS>',topn=5))
print(model.wv.most_similar(positive='<EOS>',topn=5))
save_path = r"D:\Backup\ml_data\GitHub\chat-robot_seq2seq\word2vec_model\word2vec_model"
model.save(save_path)

