import torch
import torch.nn as nn

# make program run on cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length)
        return nn.functional.cross_entropy(self.logits(x), y)


unique_chars = list(set(" hello world"))
char_encodings = [ [0]*len(unique_chars) for _ in unique_chars]
for i in range(len(unique_chars)):
    char_encodings[i][i] = 1.0

char_to_index = {char: index for index, char in enumerate(unique_chars)}
index_to_char = {index: char for char, index in char_to_index.items()}
encoding_size = len(char_encodings)

train_string = " hello world "
x_train = torch.tensor([[char_encodings[char_to_index[ch]]] for ch in train_string[:-1]], dtype=torch.float)
y_train = torch.tensor([char_to_index[ch] for ch in train_string[1:]], dtype=torch.long)


model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.0001)
for epoch in range(1000):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        # Generate characters from the initial characters ' h'
        model.reset()

        # This part is new
        text = ' h'
        seed_char = ' '
        model.f(torch.tensor([[char_encodings[char_to_index[seed_char]]]], dtype=torch.float))
        y = model.f(torch.tensor([[char_encodings[char_to_index['h']]]], dtype=torch.float))
        next_char = index_to_char[y.argmax(1).item()]
        text += next_char
        for c in range(50):
            y = model.f(torch.tensor([[char_encodings[char_to_index[next_char]]]], dtype=torch.float))
            next_char = index_to_char[y.argmax(1).item()]
            text += next_char
        print(text)