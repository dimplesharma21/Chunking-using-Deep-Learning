from gensim.models import Word2Vec
import numpy as np
from pprint import *
from collections import OrderedDict
import pickle
 
from sklearn.model_selection import train_test_split
from keras.layers import Dense,TimeDistributed
from keras.models import Sequential
from keras import layers
from keras.utils.vis_utils import plot_model
 
np.set_printoptions(threshold=np.nan)
 
def sentence_list_creation(file):
    with open(file,'r',encoding='UTF-8') as pos_file:
        lines = [line for line in pos_file.readlines()]
        empty_ind = []
        for ind, new_line_index in enumerate(lines):
            if new_line_index == '\n':
                empty_ind.append(ind)
 
    initial_indx = 0
    sentences_list = []
    for emp_ind in empty_ind:
        sentences_list.append(lines[initial_indx:emp_ind])
        initial_indx = emp_ind + 1
    return sentences_list

def word_tag_individual_list(file_name):
    sentences_list = sentence_list_creation(file_name)
    words = []
    tags = []
    chunks= []
    for sentence_id in sentences_list:
        word = []
        tag = []
        chunk=[]
        for words_with_tag in sentence_id:
            if str(words_with_tag.split(' ')[0]).startswith('\ufeff'):
                word.append(words_with_tag.split(' ')[0].lstrip('\ufeff'))
            else:
                word.append(words_with_tag.split(' ')[0])
            tag.append(words_with_tag.split(' ')[1])
            chunk.append(words_with_tag.split(' ')[2].rstrip('\n'))
        words.append(word)
        tags.append(tag)
        chunks.append(chunk)
return (words, tags, chunks)
 
def unique_tags(tag_list):
    tag = list(set([word_tag for sent_tag in tag_list for word_tag in sent_tag]))
    print (tag)
    return tag
 
def unique_chunks(chunk_list):
    chunk = list(set([word_chunk for sent_chunk in chunk_list for word_chunk in sent_chunk]))
    print (chunk)
    return chunk

def word2Vec_model(words, embedding_size):
    word_vec_model = Word2Vec(sentences=words, size=embedding_size, window=3, workers=5, min_count=1)
word_vec_model.save('train_word2vec')
return word_vec_model
 
def tag_maping_model(tags, one_hot_size):
    tags.append('NAN')
    one_hot_vector = np.zeros(one_hot_size, dtype=np.int32)
    tag_Hot_mapping = OrderedDict()
    index_tag_list = []
 
    for t_index, tag in enumerate(tags):
        if tag == 'NAN':
            index_tag_list.append((0, tag))
            tag_Hot_mapping.update({'NAN':one_hot_vector})
        else:
            index_tag_list.append((t_index+1, tag))
            temp_one_hot_vec = np.zeros(one_hot_size, dtype=np.int32)
            temp_one_hot_vec[t_index] = 1
            tag_Hot_mapping.update({tag:temp_one_hot_vec})
    index_tag_mapping = OrderedDict(index_tag_list)
return [tag_Hot_mapping, index_tag_mapping]
 
def chunk_maping_model(chunks, one_hot_size):
    chunks.append('NAN')
    one_hot_vector = np.zeros(one_hot_size, dtype=np.int32)
    chunk_Hot_mapping = OrderedDict()
    index_chunk_list = []
 
    for c_index, chunk in enumerate(chunks):
        if chunk == 'NAN':
            index_chunk_list.append((0, chunk))
            chunk_Hot_mapping.update({'NAN':one_hot_vector})
        else:
            index_chunk_list.append((c_index+1, chunk))
            temp_one_hot_vec = np.zeros(one_hot_size, dtype=np.int32)
            temp_one_hot_vec[c_index] = 1
            chunk_Hot_mapping.update({chunk:temp_one_hot_vec})
    index_chunk_mapping = OrderedDict(index_chunk_list)
    return [chunk_Hot_mapping, index_chunk_mapping]

def tag_vector(tag_Hot_mapping, tag):
    return tag_Hot_mapping[tag]
 
def chunk_vector(chunk_Hot_mapping, chunk):
    return chunk_Hot_mapping[chunk]
 
def padding_appender(word_vec, tag_vec, chunk_vec, tc):
    size_sent = max([len(size) for size in word_vec])
    for sent in word_vec:
        while(len(sent)!=size_sent):
            padding = np.zeros(tc, dtype=np.int32)
            sent.append(padding)
    for tag in tag_vec:
        while(len(tag)!=size_sent):
            padding = np.zeros(tc, dtype=np.int32)
            tag.append(padding)
 for chunk in chunk_vec:
        while(len(chunk)!=size_sent):
            padding = np.zeros(tc, dtype=np.int32)
            chunk.append(padding)
    return word_vec, tag_vec, chunk_vec
def print_out_tag(predicted_tag_list, tag_val, word_val, act_function, ep):
    pred_tag = []
    for sentence in predicted_tag_list:
        pred_tag_sent= []
        for word in sentence:
            word_ = word.tolist()
            index_ = word_.index(max(word))+1
            p_tag = index_tag_mapping[index_]
            pred_tag_sent.append(p_tag)
        pred_tag.append(pred_tag_sent)
 
    act_tag = []
    for sentence_1 in tag_val:
        act_tag_sent = []
        for tag_ in sentence_1:
            new_tag = tag_.tolist()
            ind_ = new_tag.index(max(new_tag))+1
            a_tag = index_tag_mapping[ind_]
            act_tag_sent.append(a_tag)
        act_tag.append(act_tag_sent)

actual_word = []
    for sentence_2 in word_val:
        act_word_sent = []
        for word_ in sentence_2:
            if np.array_equal(word_, np.zeros(total_class_tag, dtype=np.int32)):
                pass
            else:
                actual_words = word_vec_model.most_similar(positive=[word_], topn=1)
                act_word_sent.append(actual_words[0][0])
        actual_word.append(act_word_sent)
file_name = 'pred_out_'+str(ep)+'_'+act_function+'.txt'
    with open(file_name,'w') as output_file:
        for i in range(len(actual_word)):
            for j in range(len(actual_word[i])):
                w = actual_word[i][j]
                pt = pred_tag[i][j]
                at = act_tag[i][j]
                output_file.write(w)
                output_file.write('\t')
                output_file.write(at)
                output_file.write('\t')
                output_file.write(pt)
                output_file.write('\n')
            output_file.write('\n')
    print('\n*%s file tag created sucessfully.*' %file_name)

def print_out_chunk(predicted_chunk_list, chunk_val, word_val, act_function, ep):
    pred_chunk = []
    for sentence in predicted_chunk_list:
        pred_chunk_sent = []
        for word in sentence:
            word_ = word.tolist()
            index_ = word_.index(max(word)) + 1
            p_chunk = index_chunk_mapping[index_]
            pred_chunk_sent.append(p_chunk)
        pred_chunk.append(pred_chunk_sent)
  
    act_chunk = []
    for sentence_0 in chunk_val:
        act_chunk_sent = []
        for chunk_ in sentence_0:
            new_chunk = chunk_.tolist()
            ind_ = new_chunk.index(max(new_chunk)) + 1
            a_chunk = index_chunk_mapping[ind_]
            act_chunk_sent.append(a_chunk)
        act_chunk.append(act_chunk_sent)
  
actual_word = []
    for sentence_2 in word_val:
        act_word_sent = []
        for word_ in sentence_2:
            if np.array_equal(word_, np.zeros(total_class_chunk, dtype=np.int32)):
                pass
            else:
                actual_words = word_vec_model.most_similar(positive=[word_], topn=1)
                act_word_sent.append(actual_words[0][0])
        actual_word.append(act_word_sent)
file_name = 'pred_out_'+str(ep)+'_'+act_function+'.txt'
    with open(file_name,'w') as output_file:
        for i in range(len(actual_word)):
            for j in range(len(actual_word[i])):
                w = actual_word[i][j]
                pc= pred_chunk[i][j]
                ac= act_chunk[i][j]
                output_file.write(w)
                output_file.write('\t')
                output_file.write(ac)
                output_file.write('\t')
                output_file.write(pc)
                output_file.write('\n')
            output_file.write('\n')
    print('\n*%s file chunk created sucessfully.*' %file_name)
"""
 
Required Format ---------
word1.1		tag	chunk
word1.2		tag	chunk
word1.3		tag	chunk
 
word 2.1		tag	chunk
word2.2		tag	chunk
"""

if __name__ == '__main__':
    file_name = '/home/dimple21/PycharmProjects/test/train.txt'
    word_list, tag_list, chunk_list = word_tag_individual_list(file_name)
 
    unique_tag = unique_tags(tag_list)
    unique_chunk = unique_chunks(chunk_list)
    total_class_tag = embedding_size = len(unique_tag)+1
    total_class_chunk = embedding_size = len(unique_chunk)+1
    total_class_both=len(unique_tag)+len(unique_chunk)+1
 
    wv_model = word2Vec_model(word_list, embedding_size)
    tag_Hot_mapping, index_tag_mapping = tag_maping_model(unique_tag, total_class_tag)
    chunk_Hot_mapping, index_chunk_mapping = chunk_maping_model(unique_chunk, total_class_chunk)
 
    word_one_hot_list = []
    for word_sentence in word_list:
        sentence_word = []
        for word in word_sentence:
            sentence_word.append(wv_model[word])
        word_one_hot_list.append(sentence_word)
 
 
  tag_one_hot_list = []
    for tag_sentence in tag_list:
        sentence_tag = []
        for tag in tag_sentence:
            v_tag = tag_vector(tag_Hot_mapping, tag)
            sentence_tag.append(v_tag)
        tag_one_hot_list.append(sentence_tag)
chunk_one_hot_list = []
    for chunk_sentence in chunk_list:
        sentence_chunk = []
        for chunk in chunk_sentence:
            v_chunk = chunk_vector(chunk_Hot_mapping, chunk)
            sentence_chunk.append(v_chunk)
        chunk_one_hot_list.append(sentence_chunk)
 
    word_sent_list, tag_sent_list, chunk_sent_list= padding_appender(word_one_hot_list, tag_one_hot_list, chunk_one_hot_list, total_class_tag)
    print(np.shape(word_sent_list))
    print(np.shape(tag_sent_list))
    print(np.shape(chunk_sent_list))

word, word_test, tag, tag_test, chunk, chunk_test = train_test_split(word_sent_list, tag_sent_list, chunk_sent_list, test_size=0.1, random_state=1)
    word_train, word_val, tag_train, tag_val, chunk_train, chunk_val = train_test_split(word, tag, chunk, test_size=0.1, random_state=1)
    word_vec_model = Word2Vec.load('train_word2vec')
    hidden_layer = np.shape(word_train)[0]                 
    time_steps = np.shape(word_train)[0]
    word_dim = np.shape(word_train)[1]
    RNN = layers.LSTM
    activation_function = ['softsign','tanh','sigmoid','hard_sigmoid']
epochs = [100,200,300]

for ep in epochs:
        for act_fun in activation_function:
            model = Sequential()
            model.add(RNN(units=hidden_layer, input_shape=(time_steps, word_dim), dropout=0.5,
                           return_sequences= True, recurrent_dropout=0.5, activation=act_fun))
            model.add(TimeDistributed(Dense(units=total_class_tag, activation='softmax', kernel_initializer='normal')))
 
            model.compile(loss='categorical_crossentropy', 	metrics=['accuracy'], optimizer='adam')
            model.fit(word_train, tag_train, batch_size=15, 	verbose=2, validation_data=(word_test, 	tag_test, chunk_test), epochs=ep)
            score, acc = model.evaluate(word_test, tag_test, 	chunk_test)
            print(acc*100)
 
            predicted_tag_list = model.predict(tag_val)
            print_out_tag(predicted_tag_list, tag_val, 	word_val, act_fun, ep)
            predicted_chunk_list = model.predict(chunk_val)
            print_out_chunk(predicted_chunk_list, chunk_val, 	word_val, act_fun, ep)
	
           

