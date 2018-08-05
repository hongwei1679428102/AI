import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


raw_text = open('corpus_text.txt').read()
raw_text = raw_text.lower()

# print(raw_text)

chars = sorted(list(set(raw_text)))
print(chars)
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# print(char_to_int)
# print(int_to_char)

print(len(chars))
print(len(raw_text))

# 构造训练序列
seq_length = 100
x = []
y = []
for i in range(0, len(raw_text) - seq_length):
    given = raw_text[i: i+seq_length]
    predict = raw_text[i+seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

# print(x[:3])
# print(y[:3])

n_patterns = len(x)
n_vocab = len(chars)

# 把X变成lstm想要的样子 【样本数， 时间步伐， 特征】[多少句话， 一句话多少个词， 一个词多少个向量]
x = np.reshape(x, (n_patterns, seq_length, 1))
x = x / float(n_vocab)

# 输出变成one-hot
y = np_utils.to_categorical(y)
# print(y[:3])

# 一切准备就绪，构造模型
model = Sequential()
# 每次128个神经元
model.add(LSTM(128, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x, y, nb_epoch=1, batch_size=32)


def predict_next(input_array):
    x = np.reshape(input_array, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y


def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input) - seq_length):]:
        res.append(char_to_int[c])
    return res


def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c


def generate_article(init, rounds=500):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string


if __name__ == '__main__':
    init = """
    Ben stooped to pick up the bags. He had got hold ofEven though he
    """
    article = generate_article(init)
    print(article)


