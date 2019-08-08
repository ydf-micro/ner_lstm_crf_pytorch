# *_*coding:utf-8 *_*

import pickle
import codecs
import copy

class Data_preprocess():
    def __init__(self, batch_size=64, data_type='train'):
        self.batch_size = batch_size
        self.data_type = data_type
        self.data = []
        self.batch_data =  []
        self.vocab = {'unk': 0}
        self.tags_map = {'<START>': 0, '<STOP>': 1}

        if data_type == 'train':
            self.data_path = '../data/example.train'
        elif data_type == 'dev':
            self.data_path = '../data/example.dev'
            self.load_data_map()
        elif data_type == 'test':
            self.data_path = '../data/example.test'
            self.load_data_map()

        self.load_data()
        self.prepare_batch()

    def load_data_map(self):
        with codecs.open('../data/data.pkl', 'rb') as f:
            self.data_map = pickle.load(f)
            self.vocab = self.data_map.get('vocab', {})
            self.tags_map = self.data_map.get('tags_map', {})
            self.tags = self.data_map.keys()

    def load_data(self):
        sentence = []
        target = []

        with codecs.open(self.data_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    self.data.append([sentence, target])
                    sentence = []
                    target = []
                    continue

                word, tag = line.split()
                if word not in self.vocab and self.data_type == 'train':
                    self.vocab[word] = len(self.vocab)
                if tag not in self.tags_map and self.data_type == 'train':
                    self.tags_map[tag] = len(self.tags_map)
                sentence.append(self.vocab.get(word, 0))
                target.append(self.tags_map.get(tag, 0))

        print(self.tags_map)

        self.input_size = len(self.vocab)
        print(f'{self.data_type} data: {len(self.data)}')
        print(f'vocab size: {self.input_size}')
        print(f'unique tag: {len(self.tags_map)}')
        print('*'*50)

    def prepare_batch(self):
        index = 0
        while True:
            if index + self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index: index+self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)


    def pad_data(self, data):
        c_data = copy.deepcopy(data)
        max_length = max([len(i[0]) for i in c_data])
        for c in c_data:
            c.append(len(c[0]))
            c[0] += (max_length - len(c[0])) * [0]
            c[1] += (max_length - len(c[1])) * [0]

        return c_data

    def iteration(self):
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data)-1:
                idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data
