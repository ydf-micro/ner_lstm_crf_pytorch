# *_*coding:utf-8 *_*

import pickle
import torch
import codecs
import torch.optim as optim
from data_preprocess import Data_preprocess
from model import BiLSTM_CRF
from utils import f1_score


BATCH_SIZE = 128
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 128
DROPOUT = 1.0

class ChineseNER():
    def __init__(self, entry='train'):
        if entry == 'train':
            self.train_manager = Data_preprocess(batch_size=BATCH_SIZE)
            self.total_size = len(self.train_manager.batch_data)
            data = {
                'batch_size': self.train_manager.batch_size,
                'input_size': self.train_manager.input_size,
                'vocab': self.train_manager.vocab,
                'tags_map': self.train_manager.tags_map
            }
            self.save_params(data)
            dev_manager = Data_preprocess(batch_size=BATCH_SIZE, data_type='dev')
            self.dev_batch = dev_manager.iteration()

            self.model = BiLSTM_CRF(
                vocab_size=len(self.train_manager.vocab),
                tag_to_ix=self.train_manager.tags_map,
                embedding_dim=EMBEDDING_SIZE,
                hidden_dim=HIDDEN_SIZE,
                batch_size=BATCH_SIZE
            )
            self.model.cuda()

        elif entry == 'predict':
            data_map = self.load_params()
            input_size = data_map.get('input_size')
            self.tags_map = data_map.get('tags_map')
            self.vocab = data_map.get('vocab')

            self.model = BiLSTM_CRF(
                vocab_size=input_size,
                tag_to_ix=self.tags_map,
                embedding_dim=EMBEDDING_SIZE,
                hidden_dim=HIDDEN_SIZE
            )
            self.model.cuda()
            self.restore_model()

    def save_params(self, data):
        with codecs.open('../data/data.pkl', 'wb') as f:
            pickle.dump(data, f)

    def restore_model(self):
        self.model.load_state_dict(torch.load('../data/params.pkl'))

        print('model restore success!')

    def load_params(self):
        with codecs.open('../data/data.pkl', 'rb') as f:
            data_map = pickle.load(f)

        return data_map

    def train(self):
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(50):
            index = 0
            for batch in self.train_manager.get_batch():
                index += 1
                self.model.zero_grad()

                sentences, tags, length = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long).cuda()
                tags_tensor = torch.tensor(tags, dtype=torch.long).cuda()
                length_tensor = torch.tensor(length, dtype=torch.long).cuda()

                loss = self.model.neg_log_likelihood(sentences_tensor,
                                                     tags_tensor,
                                                     length_tensor)
                progress = ('█' * int(index * 25 / self.total_size)).ljust(25)
                print(f'epoch [{epoch+1}] |{progress}| '
                      f'{index}/{self.total_size}\n\t'
                      f'loss {loss.cpu().tolist()[0]:.2f}')

                print('-'*50)
                loss.backward()
                optimizer.step()

            self.evaluate()
            print('*' * 50)
            torch.save(self.model.state_dict(), '../data/params.pkl')

    def evaluate(self):
        sentences, tags, lengths = zip(*self.dev_batch.__next__())
        _, paths = self.model(sentences, lengths)
        print('\tevaluation')
        f1_score(tags, paths, lengths)

    def predict(self, input_str=''):
        ix_to_ner = {v: k for k, v in self.tags_map.items()}
        if not input_str:
            input_str = input('请输入文本:')
        input_vec = [self.vocab.get(str, 0) for str in input_str]
        sentences = torch.tensor(input_vec).view(1, -1).cuda()
        _, paths = self.model(sentences, [sentences.size(1)])

        res = []
        for path in paths[0]:
            res.append(ix_to_ner[path])

        print(input_str)
        print(res)


if __name__ == '__main__':
    type = input('train or predict\n')
    # type = 'train'
    if type == 'train':
        ner = ChineseNER('train')
        ner.train()
    elif type == 'predict':
        ner = ChineseNER('predict')
        ner.predict()