# *_*coding:utf-8 *_*
import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"

# compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):

    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)

    return result.squeeze(1)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix,
                 embedding_dim=100, hidden_dim=128,
                 batch_size=64):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2,
                            num_layers=1, bidirectional=True,
                            batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # Matrix of transition parameters. Entry i,j is
        # the score of transitioning from i to j
        self.transitions = nn.Parameter(torch.randn(self.tagset_size,
                                                    self.tagset_size))

        # These two statements enforce the constraint
        # that we never transfer to the start tag and
        # we never transfer from the stop tag
        self.transitions.data[:, self.tag_to_ix[START_TAG]] = -10000.
        self.transitions.data[self.tag_to_ix[STOP_TAG], :] = -10000.

        self.hidden = self.init_hidden()

    def init_hidden(self):

        return (torch.randn(2, self.batch_size, self.hidden_dim//2).cuda(),
                torch.randn(2, self.batch_size, self.hidden_dim//2).cuda())

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        seq_len = sentence.size(1)
        embeds = self.word_embeds(sentence).view(self.batch_size, seq_len, self.embedding_dim)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _forward_alg(self, emissions):

        previous = torch.full((1, self.tagset_size), 0).cuda()

        for index in range(len(emissions)):
            previous = torch.transpose(previous.expand(self.tagset_size, self.tagset_size), 0, 1)
            obs = emissions[index].view(1, -1).expand(self.tagset_size, self.tagset_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)

        previous = previous + self.transitions[:, self.tag_to_ix[STOP_TAG]]

        # calculate total_scores
        total_scores = log_sum_exp(torch.transpose(previous, 0, 1))[0]

        return total_scores

    def _score_sentences(self, emissions, tags):
        # Gives the score of a provided tag sequence
        # Score = Emission_Score + Transition_Score
        score = torch.zeros(1).cuda()
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).cuda(), tags])
        for i, emission in enumerate(emissions):
            score += self.transitions[tags[i], tags[i+1]] + emission[tags[i+1]]
        score += self.transitions[tags[-1], self.tag_to_ix[STOP_TAG]]

        return score

    def neg_log_likelihood(self, sentences, tags, length):
        self.batch_size = sentences.size(0)
        emissions = self._get_lstm_features(sentences)
        gold_score = torch.zeros(1).cuda()
        total_score = torch.zeros(1).cuda()
        for emission, tag, len in zip(emissions, tags, length):
            emission = emission[:len]
            tag = tag[:len]
            gold_score += self._score_sentences(emission, tag)
            total_score += self._forward_alg(emission)

        return (total_score - gold_score) / self.batch_size

    def _viterbi_decode(self, emissions):
        trellis = torch.zeros(emissions.size()).cuda()
        backpointers = torch.zeros(emissions.size(), dtype=torch.long).cuda()

        trellis[0] = emissions[0]
        for t in range(1, len(emissions)):
            v = trellis[t-1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = emissions[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.cpu().numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()

        return viterbi_score, viterbi

    def forward(self, sentences, lengths=None):

        sentence = torch.tensor(sentences, dtype=torch.long).cuda()
        if not lengths:
            lengths = [sen.size(-1) for sen in sentence]
        self.batch_size = sentence.size(0)

        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)

        scores = []
        paths = []
        for emission, len in zip(emissions, lengths):
            emission = emission[:len]
            score, path = self._viterbi_decode(emission)
            scores.append(score)
            paths.append(path)

        return scores, paths