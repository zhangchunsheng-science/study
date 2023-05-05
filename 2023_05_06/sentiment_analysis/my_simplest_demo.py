from collections import Counter
from string import punctuation

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

############################################################################
# 1. work with data
############################################################################

# 1.0
# download data
#
# https://www.kaggle.com/datasets/ehabibrahim758/dataset?select=reviews.txt
# https://www.kaggle.com/datasets/ehabibrahim758/dataset?select=labels.txt


# 1.1 read data
with open('data/reviews.txt', 'r') as f:
    reviews = f.readlines()
with open('data/labels.txt', 'r') as f:
    labels = f.readlines()

# 1.2 punctuation
print(punctuation)

# 1.3 remove punctuation and split to words
all_reviews = list()
for text in reviews:
    text = text.lower()
    text = "".join([ch for ch in text if ch not in punctuation])
    all_reviews.append(text)
all_text = " ".join(all_reviews)
all_words = all_text.split()

count_words = Counter(all_words)
total_words = len(all_words)
sorted_words = count_words.most_common(total_words)
print("Top ten occuring words : ")
print(sorted_words[:10])

# 1.4 make a map, key is word and value is its index
vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}

# 1.5 map all_reviews to a list
#     each element of the list is the value of the word in vocab_to_int
encoded_reviews = list()
for review in all_reviews:
    encoded_review = list()
    for word in review.split():
        if word not in vocab_to_int.keys():
            # if word is not available in vocab_to_int put 0 in that place
            encoded_review.append(0)
        else:
            encoded_review.append(vocab_to_int[word])
    encoded_reviews.append(encoded_review)

# 1.6 make the encoded_reviews same length
sequence_length = 250
features = np.zeros((len(encoded_reviews), sequence_length), dtype=int)
for i, review in enumerate(encoded_reviews):
    review_len = len(review)
    if (review_len <= sequence_length):
        zeros = list(np.zeros(sequence_length - review_len))
        new = zeros + review
    else:
        new = review[:sequence_length]
    features[i, :] = np.array(new)

# 1.5 handle labels
labels = [1 if label.strip() == 'positive' else 0 for label in labels]

# 1.6 split train and test datasets
train_x = features[:int(0.8 * len(features))]
train_y = labels[:int(0.8 * len(features))]
valid_x = features[int(0.8 * len(features)):int(0.9 * len(features))]
valid_y = labels[int(0.8 * len(features)):int(0.9 * len(features))]
test_x = features[int(0.9 * len(features)):]
test_y = labels[int(0.9 * len(features)):]
print(len(train_y), len(valid_y), len(test_y))

# 1.7 create Tensor Dataset
train_data = TensorDataset(torch.LongTensor(train_x), torch.LongTensor(train_y))
valid_data = TensorDataset(torch.LongTensor(valid_x), torch.LongTensor(valid_y))
test_data = TensorDataset(torch.LongTensor(test_x), torch.LongTensor(test_y))

# 1.8 dataloader
batch_size = 50
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


############################################################################
# 2. creating models
############################################################################


class SentimentalLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers
        """
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # Linear and sigmoid layer
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        # x (batch_size, sequence_length)
        # hidden ((n_layers * directions, batch_size, hidden_dim), (n_layers * directions, batch_size, hidden_dim))
        batch_size = x.size()

        # Embadding and LSTM output
        # embedd (batch_size, sequence_length, embedding_dim)
        embedd = self.embedding(x)
        # lstm_out (batch_size, sequence_length, hidden_dim)
        # hidden ((n_layers * directions, batch_size, hidden_dim), (n_layers * directions, batch_size, hidden_dim))
        lstm_out, hidden = self.lstm(embedd, hidden)

        # stack up the lstm output
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)  # lstm_out (batch_size * sequence_length, hidden_dim)

        # dropout and fully connected layers
        out = self.dropout(lstm_out)  # out (batch_size * sequence_length, hidden_dim)
        out = self.fc1(out)           # out (batch_size * sequence_length, 64)
        out = self.dropout(out)       # out (batch_size * sequence_length, 64)
        out = self.fc2(out)           # out (batch_size * sequence_length, 16)
        out = self.dropout(out)       # out (batch_size * sequence_length, 16)
        out = self.fc3(out)           # out (batch_size * sequence_length, 1)
        sig_out = self.sigmoid(out)   # sig_out (batch_size * sequence_length, 1)

        sig_out = sig_out.view(batch_size, -1)  # sig_out (batch_size, sequence_length)
        sig_out = sig_out[:, -1]                # sig_out (batch_size,)

        return sig_out, hidden

    def init_hidden(self, batch_size):
        """Initialize Hidden STATE"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentalLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(net)

# loss and optimization functions
lr = 0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params
epochs = 3  # 3-4 is approx where I noticed the validation loss stop decreasing
counter = 0
print_every = 100
clip = 5  # gradient clipping

############################################################################
# 3. optimizing model parameters
############################################################################


net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

test_losses = []  # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)

# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

"""
Output

!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
Top ten occuring words : 
[('the', 336713), ('and', 164107), ('a', 163009), ('of', 145864), ('to', 135720), ('is', 107328), ('br', 101872), ('it', 96352), ('in', 93968), ('i', 87623)]
20000 2500 2500
SentimentalLSTM(
  (embedding): Embedding(74073, 400)
  (lstm): LSTM(400, 256, num_layers=2, batch_first=True, dropout=0.5)
  (dropout): Dropout(p=0.3, inplace=False)
  (fc1): Linear(in_features=256, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=16, bias=True)
  (fc3): Linear(in_features=16, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
Epoch: 1/3... Step: 100... Loss: 0.653277... Val Loss: 0.635075
Epoch: 1/3... Step: 200... Loss: 0.711719... Val Loss: 0.657688
Epoch: 1/3... Step: 300... Loss: 0.710229... Val Loss: 0.666137
Epoch: 1/3... Step: 400... Loss: 0.484649... Val Loss: 0.515293
Epoch: 2/3... Step: 500... Loss: 0.606124... Val Loss: 0.508251
Epoch: 2/3... Step: 600... Loss: 0.469886... Val Loss: 0.450613
Epoch: 2/3... Step: 700... Loss: 0.462995... Val Loss: 0.422026
Epoch: 2/3... Step: 800... Loss: 0.229011... Val Loss: 0.385836
Epoch: 3/3... Step: 900... Loss: 0.253350... Val Loss: 0.415577
Epoch: 3/3... Step: 1000... Loss: 0.267610... Val Loss: 0.440173
Epoch: 3/3... Step: 1100... Loss: 0.295505... Val Loss: 0.397089
Epoch: 3/3... Step: 1200... Loss: 0.096462... Val Loss: 0.406767
Test loss: 0.414
Test accuracy: 0.818
"""
