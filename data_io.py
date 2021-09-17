import pandas as pd
import numpy as np
from konlpy.tag import Okt
from gensim.models import fasttext, FastText#ver-4.0.1
import re
import os
from PIL import Image
from model import *
import json

path = './'

ddata = path+'data/sample_ddata.txt'
mdata = path+'data/mdata.txt'
tdata = path+'data/ac_eval_t1.wst.dev'


### create data list ###

def leave_only_korean(sentence, do_split=True):
    if do_split: return re.compile('[ㄱ_ㅎ|ㅏ-ㅣ|가-힣]+').findall(sentence)
    return re.sub(r'[^ ㄱ-ㅣ가-힣]', '', sentence)

def remove_stop_word(sentence):
    with open('model_file/stopword.txt', encoding='utf-8', mode='r') as f:
        stopword = f.read().split()
    res = []
    for word in sentence:
        if word not in stopword:
            res.append(word)
    return(res)


def sentence_processing(sentence):
    okt = Okt()
    processed_sentence = []
    for word in okt.pos(sentence, stem=True):
        if word[1] in ['Noun', 'Verb', 'Adjective']:
            processed_sentence.append(word[0])
    return processed_sentence


def get_data_list(data_type = 'train'):
    dia_index_list = []
    dia_code_list = []
    dia_data_list = []
    image_list = []
    dia_tag_list = []
    if data_type == 'train': data_path = ddata
    elif data_type == 'test':
        data_path = tdata
        idx = 0
    with open(data_path, encoding='euc-kr', mode='r') as f:
        data = []
        count = 0 ###################
        for l in f.readlines():
            dia_index = []
            dia_code = []
            dia_data = []
            image = []
            dia_tag = []
            l = l.replace('_', '').replace('<', '').replace('>', '')

            count = count+1##############################
            if data_type == 'train':
                if len(l.split())>0:
                    dia_index = l.split()[0]
                    if len(l.split())>1: dia_code = l.split()[1]
                    elif l.split()[0][-1].encode().isalpha(): dia_tag_list[-1] = l.split()[0]
                    if l.split()[-1][-1].encode().isalpha(): dia_tag = l.split()[-1]

                if dia_code == 'AC':
                    image = l.split()[2:]
            elif data_type == 'test':
                if l.split()[0] != ';':
                    idx = idx+1
                    dia_index = idx
                    dia_code = l.split()[0]
                    if dia_code[0] == 'R':
                        image = l.split()[1:]
                else:
                    idx = 0
            if data_type == 'train' or idx != 0:
                #dia_data = leave_only_korean(l, do_split=False)
                #dia_data = sentence_processing(dia_data)
                dia_data = leave_only_korean(l, do_split=True)

            dia_data = remove_stop_word(dia_data)

            dia_index_list.append(dia_index)
            dia_code_list.append(dia_code)
            dia_data_list.append(dia_data)
            image_list.append(image)
            dia_tag_list.append(dia_tag)

    return dia_index_list, dia_code_list, dia_data_list, image_list, dia_tag_list

def get_max_length(dia_data):
    max_len = 0
    for sentence in dia_data:
        if max_len<len(sentence):
            max_len = len(sentence)
    return max_len

def get_base_embedding_model(base_model = path+'model_file/cc.ko.300.bin'):
    pre_trained_model = fasttext.load_facebook_model(path+'model_file/cc.ko.300.bin')
    return pre_trained_model

def train_embedding_model(model, dia_data):
    model.train(dia_data, total_examples=len(dia_data), epochs=10)

def save_embedding_model(model):
    model.wv.save(path+'model_file/ko_embVec')

def load_embedding_model():
    return fasttext.FastTextKeyedVectors.load(path+'model_file/ko_embVec')

def make_embedding_model(dia_data, base_model_path=None, emb_size=300, win_size=3):
    if base_model_path:
        model = get_base_embedding_model(base_model_path)
        train_embedding_model(model, dia_data)
    else:
        model = FastText(dia_data, vector_size=emb_size, window=win_size, min_count=0, workers=1, sg=1)
    model.init_sims(replace=True)
    save_embedding_model(model)

def get_embedding_sentence(sentence, embedding_model):
    embedding_sentence = []
    for word in sentence:
        embedding_word = embedding_model[word]
        embedding_sentence.append(embedding_word)
    return embedding_sentence
def get_embedding_data(dia_data, max_len, emb_size, embedding_model):
    num_data = len(dia_data)
    embedding_data = np.zeros((num_data, max_len, emb_size))
    for idx in range(num_data):
        sentence = dia_data[idx]
        if sentence:
            embedding_sentence = np.array(get_embedding_sentence(sentence, embedding_model))
            embedding_data[idx][0:embedding_sentence.shape[0]] = embedding_sentence
    return embedding_data

def padding(dia_emb_list, emb_size, max_len):
    zero_emb = [0.]*emb_size
    zero_emb = np.array(zero_emb)
    padded_emb_list = []
    for dia_emb in dia_emb_list:
        pad = max_len - len(dia_emb)
        for i in range(pad):
            dia_emb.append(zero_emb)
        padded_emb_list.append(dia_emb)
    return padded_emb_list

def make_img_metaData_file(embedding_model, emb_size, meta_data_path=mdata):
    img_name_list = []
    item_list = []
    item_detail_list = []
    cha_list = []
    cha_data_list = []
    temp_cha_emb = [0.]*emb_size
    cha_emb = [None, None, None, None]
    with open(meta_data_path, encoding='euc-kr', mode='r') as f:
        for l in f.readlines():
            img_name = l.split()[0]
            item = l.split()[1]
            item_detail = l.split()[2]
            cha = l.split()[3]
            #cha_data = leave_only_korean(l, do_split=False)
            #cha_data = sentence_processing(cha_data)
            cha_data = leave_only_korean(l, do_split=True)
            cha_data = remove_stop_word(cha_data)

            if img_name_list==[]:
                img_name_list.append(img_name)
                item_list.append(item)
                item_detail_list.append(item_detail)

            else:
                if cha != cha_list[-1]:
                    if cha_list[-1] == 'F':
                        cha_emb[0] = temp_cha_emb.tolist()
                    elif cha_list[-1] == 'M':
                        cha_emb[1] = temp_cha_emb.tolist()
                    elif cha_list[-1] == 'C':
                        cha_emb[2] = temp_cha_emb.tolist()
                    elif cha_list[-1] == 'E':
                        cha_emb[3] = temp_cha_emb.tolist()
                    temp_cha_emb = [0.]*emb_size

                    if img_name != img_name_list[-1]:
                        img_name_list.append(img_name)
                        item_list.append(item)
                        item_detail_list.append(item_detail)
                        cha_list.append(cha)
                        cha_data_list.append(cha_emb)
                        cha_emb = [None, None, None, None]

            for chad in cha_data:
                temp_cha_emb = temp_cha_emb + embedding_model[chad]
            cha_list.append(cha)
        if cha_list[-1] == 'F':
            cha_emb[0] = temp_cha_emb.tolist()
        elif cha_list[-1] == 'M':
            cha_emb[1] = temp_cha_emb.tolist()
        elif cha_list[-1] == 'C':
            cha_emb[2] = temp_cha_emb.tolist()
        elif cha_list[-1] == 'E':
            cha_emb[3] = temp_cha_emb.tolist()
        cha_data_list.append(cha_emb)
        df = pd.DataFrame({'img_name': img_name_list, \
                           'item': item_list, \
                           'item_detail': item_detail_list, \
                           'character_emb': cha_data_list})
        df.to_csv('model_file/metaDataSet.csv')

def get_train_dataSet(dia_index_list, dia_code_list, dia_emb_list, image_list, dia_tag_list):
    idx = 0
    train_dataSet = []
    while idx<len(dia_index_list):
        story = []
        story_code = []
        query = []
        image = [None, None, None, None]
        get_US = True
        while 'CLOSING' not in dia_tag_list[idx]:
            if query == [] and dia_code_list[idx]=='US':
                query.append(dia_emb_list[idx])
            elif 'EXP' in dia_tag_list[idx] or 'SUGGEST' in dia_tag_list[idx] or 'ASK' in dia_tag_list[idx]:
                story.append(dia_emb_list[idx])
                story_code.append(dia_code_list[idx])
                get_US=True
            elif 'CONFIRM' in dia_tag_list[idx]:
                get_US=False
            elif dia_code_list[idx] == 'US' and get_US:
                story.append(dia_emb_list[idx])
                story_code.append(dia_code_list[idx])
            elif image_list[idx] != []:
                ad = 0
                while 'USER' not in dia_tag_list[idx+ad] and 'CLOSING' not in dia_tag_list[idx+ad]:
                    ad = ad+1
                if 'USERSUCCESS' in dia_tag_list[idx+ad]:
                    for img in image_list[idx]:
                        #if 'SE' in img: image[3] = img
                        cl = img[0:2]
                        if cl=='SE': image[3] = img
                        elif cl in ['SK', 'PT', 'OP']: image[2] = img
                        elif cl in ['KN', 'SW', 'SH', 'BL']: image[1] = img
                        else: image[0] = img
            idx = idx+1
        if image != [None, None, None, None]:
            train_dataSet.append([query, story_code, story, image])
        idx = idx+1
    return train_dataSet

def new_get_train_dataSet(dia_index_list, dia_code_list, dia_emb_list, image_list, dia_tag_list, story_size, sentence_size, emb_size):
    idx = 0
    train_querySet = []
    train_storySet = []
    train_imageSet = []
    while idx<len(dia_index_list):
        story = np.zeros((story_size, sentence_size, emb_size))
        story_id = 0
        query = []
        image = [None, None, None, None]
        get_US = True
        while 'CLOSING' not in dia_tag_list[idx]:
            if query == [] and dia_code_list[idx]=='US':
                query = dia_emb_list[idx]
            elif 'EXP' in dia_tag_list[idx] or 'SUGGEST' in dia_tag_list[idx] or 'ASK' in dia_tag_list[idx]:
                story[story_id] = dia_emb_list[idx]
                story_id = story_id+1
                get_US=True
            elif 'CONFIRM' in dia_tag_list[idx]:
                get_US=False
            elif dia_code_list[idx] == 'US' and get_US:
                story[story_id] = dia_emb_list[idx]
                story_id = story_id + 1
            elif image_list[idx] != []:
                ad = 0
                while 'USER' not in dia_tag_list[idx+ad] and 'CLOSING' not in dia_tag_list[idx+ad]:
                    ad = ad+1
                if 'USERSUCCESS' in dia_tag_list[idx+ad]:
                    for img in image_list[idx]:
                        #if 'SE' in img: image[3] = img
                        cl = img[0:2]
                        if cl=='SE': image[3] = img
                        elif cl in ['SK', 'PT', 'OP']: image[2] = img
                        elif cl in ['KN', 'SW', 'SH', 'BL']: image[1] = img
                        else: image[0] = img
            idx = idx+1
        if image != [None, None, None, None]:
            train_querySet.append([query])
            train_storySet.append(story)
            train_imageSet.append(image)
        idx = idx+1
    return train_querySet, train_storySet, train_imageSet

def get_test_dataSet(dia_index_list, dia_code_list, dia_emb_list, image_list):
    num_data = len(dia_index_list)
    test_dataSet = []
    story = []
    story_code = []
    query = []
    imageSet1 = [None, None, None, None]
    imageSet2 = [None, None, None, None]
    imageSet3 = [None, None, None, None]
    fir_US = False
    for idx in range(num_data):
        story_code.append(dia_code_list[idx])
        if "R" in dia_code_list[idx]:
            if dia_code_list[idx] == 'R1':
                for img in image_list[idx]:
                    cl = img[0:2]
                    if cl == 'SE':
                        imageSet1[3] = img
                    elif cl in ['SK', 'PT', 'OP']:
                        imageSet1[2] = img
                    elif cl in ['KN', 'SW', 'SH', 'BL']:
                        imageSet1[1] = img
                    else:
                        imageSet1[0] = img
            elif dia_code_list[idx] == 'R2':
                for img in image_list[idx]:
                    cl = img[0:2]
                    if cl == 'SE':
                        imageSet2[3] = img
                    elif cl in ['SK', 'PT', 'OP']:
                        imageSet2[2] = img
                    elif cl in ['KN', 'SW', 'SH', 'BL']:
                        imageSet2[1] = img
                    else:
                        imageSet2[0] = img
            elif dia_code_list[idx] == 'R3':
                for img in image_list[idx]:
                    cl = img[0:2]
                    if cl == 'SE':
                        imageSet3[3] = img
                    elif cl in ['SK', 'PT', 'OP']:
                        imageSet3[2] = img
                    elif cl in ['KN', 'SW', 'SH', 'BL']:
                        imageSet3[1] = img
                    else:
                        imageSet3[0] = img
                test_dataSet.append([query, story_code, story, imageSet1, imageSet2, imageSet3])
                story = []
                story_code = []
                query = []
                imageSet1 = [None, None, None, None]
                imageSet2 = [None, None, None, None]
                imageSet3 = [None, None, None, None]
                fir_US = False
        elif not fir_US and dia_code_list[idx] == "US":
            fir_US =True
            query = dia_emb_list[idx]
        else:
            story.append(dia_emb_list[idx])
    return test_dataSet

def new_get_test_dataSet(dia_index_list, dia_code_list, dia_emb_list, image_list, story_size, sentence_size, emb_size):
    num_data = len(dia_index_list)
    test_querySet = []
    test_storySet = []
    test_imageSet = []
    story = np.zeros((story_size, sentence_size, emb_size))
    story_id = 0
    query = None
    imageSet1 = [None, None, None, None]
    imageSet2 = [None, None, None, None]
    imageSet3 = [None, None, None, None]
    fir_US = False
    for idx in range(num_data):
        if "R" in dia_code_list[idx]:
            if dia_code_list[idx] == 'R1':
                for img in image_list[idx]:
                    cl = img[0:2]
                    if cl == 'SE':
                        imageSet1[3] = img
                    elif cl in ['SK', 'PT', 'OP']:
                        imageSet1[2] = img
                    elif cl in ['KN', 'SW', 'SH', 'BL']:
                        imageSet1[1] = img
                    else:
                        imageSet1[0] = img
            elif dia_code_list[idx] == 'R2':
                for img in image_list[idx]:
                    cl = img[0:2]
                    if cl == 'SE':
                        imageSet2[3] = img
                    elif cl in ['SK', 'PT', 'OP']:
                        imageSet2[2] = img
                    elif cl in ['KN', 'SW', 'SH', 'BL']:
                        imageSet2[1] = img
                    else:
                        imageSet2[0] = img
            elif dia_code_list[idx] == 'R3':
                for img in image_list[idx]:
                    cl = img[0:2]
                    if cl == 'SE':
                        imageSet3[3] = img
                    elif cl in ['SK', 'PT', 'OP']:
                        imageSet3[2] = img
                    elif cl in ['KN', 'SW', 'SH', 'BL']:
                        imageSet3[1] = img
                    else:
                        imageSet3[0] = img
                test_querySet.append([query])
                test_storySet.append(story)
                test_imageSet.append([imageSet1, imageSet2, imageSet3])
                story = np.zeros((story_size, sentence_size, emb_size))
                story_id = 0
                query = None
                imageSet1 = [None, None, None, None]
                imageSet2 = [None, None, None, None]
                imageSet3 = [None, None, None, None]
                fir_US = False
        elif not fir_US and dia_code_list[idx] == "US":
            fir_US = True
            query = dia_emb_list[idx]
        else:
            story[story_id] = dia_emb_list[idx]
            story_id = story_id+1
    return test_querySet, test_storySet, test_imageSet

def get_meta_data(img_name, meta_data_size, meta_data_set):
    if not img_name: return np.zeros((4, meta_data_size), dtype=np.float32)
    if '.jpg' in img_name: img_name = img_name[:-4]
    meta_data = meta_data_set.loc[meta_data_set['img_name'] == img_name]
    meta_data = meta_data['character_emb']
    meta_data = np.array(meta_data)
    return meta_data[0]

def read_padded_image(img_name, img_size=(400, 400)):
    if not img_name: return np.zeros((3, img_size[0], img_size[1]), dtype=np.float32)
    if '.jpg' not in img_name: img_name = img_name+'.jpg'
    img_path = path+'data/image/'+img_name
    img_pil = Image.open(img_path)
    img = np.array(img_pil)

    padded_img = []
    if len(img[0]) < img_size[1]:
        zero_pad = np.zeros((img_size[1] - len(img[0]), 3), dtype=int)
        for row in img:
            padded_img.append(np.concatenate((row, zero_pad)))
        padded_img = np.array(padded_img)
    else:
        padded_img = img[:, :img_size[1], :]
    if len(img) < img_size[0]:
        zero_pad = np.zeros((img_size[0] - len(img), img_size[1], 3), dtype=int)
        padded_img = np.concatenate((padded_img, zero_pad))
    else:
        padded_img = padded_img[:img_size[0], :, :]
    padded_img = np.transpose(padded_img, (2, 0, 1)).astype(np.float32)

    return padded_img

def make_img_emb_file(emb_size, img_size, meta_data_set, img_path=path+'data/image/', out_file_path=path+'model_file/imgEmbDataSet.json'):
    img_list = os.listdir(img_path)
    img_emb_model = ImageEmbeddingModel(emb_size, img_size)
    img_emb_model.load_state_dict(torch.load(path+'model_file/img_emb_model_state_dict'))
    img_emb_list = {}
    for img in img_list:
        print(img)
        img_emb_list[img] = []
        padded_img = np.array(read_padded_image(img), dtype=np.float32)
        padded_img = torch.tensor(padded_img)
        img_emb = img_emb_model(padded_img.unsqueeze(0)) #300
        img_meta = np.array(get_meta_data(img, emb_size, meta_data_set), dtype=np.float32) #(3,300)
        img_emb_list[img].append(img_emb.squeeze(0).tolist())
        img_emb_list[img].append(img_meta[0].tolist())
        img_emb_list[img].append(img_meta[1].tolist())
        img_emb_list[img].append(img_meta[2].tolist())
        img_emb_list[img].append(img_meta[3].tolist())

    with open(out_file_path, 'w') as outfile:
        json.dump(img_emb_list, outfile)





