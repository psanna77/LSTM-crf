# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import *

torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim // 2),
                torch.zeros(2, 1, self.hidden_dim // 2))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


##################### Trainer #####################
if __name__ == '__main__':
    training_data = [
    (list("apple"), ["A", "P", "P", "L", "E"]),
        (list("book"), ["B", "O", "O", "K"]),
        (list("cat"), ["C", "A", "T"]),
        (list("dog"), ["D", "O", "G"])
    ]
    char_to_ix_HRL = {}
    char_to_ix_EN = {}
    for chars_HRL, chars_EN in training_data:
        for ch in chars_HRL:
            if ch not in char_to_ix_HRL:
                char_to_ix_HRL[ch] = len(char_to_ix_HRL)

        for ch in chars_EN:
            if ch not in char_to_ix_EN:
                char_to_ix_EN[ch] = len(char_to_ix_EN)
                
    print(char_to_ix_HRL)
    print(char_to_ix_EN)

    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    EMBEDDING_DIM = 20
    HIDDEN_DIM = 20

    model_HRL = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(char_to_ix_HRL), len(char_to_ix_HRL))
    model_EN = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(char_to_ix_EN), len(char_to_ix_EN))
    
    loss_function = nn.MarginRankingLoss()
    
    optimizer_HRL = optim.SGD(model_HRL.parameters(), lr=0.1)
    optimizer_EN = optim.SGD(model_EN.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], char_to_ix_HRL)
        tag_scores = model_HRL(inputs)
        print(tag_scores)

    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for char_list_HRL, char_list_EN in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model_HRL.zero_grad()
            model_EN.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model_HRL.hidden = model_HRL.init_hidden()
            model_EN.hidden = model_EN.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            char_list_in_HRL = prepare_sequence(char_list_HRL, char_to_ix_HRL)
            char_list_in_EN = prepare_sequence(char_list_EN, char_to_ix_EN)

            random_char_list = []
            for i in range(len(char_list_HRL)):
                random_sample = training_data[randint(0, len(training_data) - 1)][1]
                random_char_list.append(random_sample[i % len(random_sample)])
            char_list_in_RAND = prepare_sequence(random_char_list, char_to_ix_EN)
            
            targets_HRL = prepare_sequence(char_list_EN, char_to_ix_EN)
            targets_EN = prepare_sequence(char_list_HRL, char_to_ix_HRL)

            # Step 3. Run our forward pass.
            tag_scores_HRL = model_HRL(char_list_in_HRL)
            tag_scores_EN = model_EN(char_list_in_EN)

            # print(tag_scores_HRL)
            rand_score = model_EN(char_list_in_RAND)
            # print(rand_score)
            
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            v_HRL = F.cosine_similarity(tag_scores_HRL, tag_scores_EN)
            # print(tag_scores_HRL)
            # print(rand_score)
            v_EN = F.cosine_similarity(tag_scores_HRL, rand_score)

            loss = loss_function(v_HRL, v_EN, torch.FloatTensor([1]))
            loss.backward()
            print(loss)
            optimizer_HRL.step()
            optimizer_EN.step()
