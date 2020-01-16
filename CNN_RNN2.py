import talib as ta
import tensorflow as tf
import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt

# 常數定義
time_step = 5
shift_max = time_step    # 承上: 一次取幾個批次資料
rnn_unit = 10            # hidden layer units
batch_size = 2           # 每一批次訓練的樣本數
input_size = 5           # 輸入層的維度
output_size = 1          # 輸出層的維度
lr = 0.0006              # 學習率

RSI_timeperiod = 6  # RSI 6天週期

DataPath = "D:\\CPRindicator\\2330\\"

f = open(DataPath + "2330A.csv")
df = pd.read_csv(f)     # 讀取 2330 行情資料
data = df.iloc[:, 1:].values

# CNN 定義
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.01))

def cnn(X, w1, w2):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME'))
    l1b = tf.nn.max_pool(l1a, strides=[1, 2, 2, 1], ksize=[
                         1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1b, 0.8)
    l2 = tf.reshape(l1, [-1, 288])           # 36 * 32 = 576 values
    return tf.matmul(l2, w2)


# RNN 定義 (權重、偏移量)
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1])),
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}

# 取得訓練資料 1
def get_train_data(batch_size=10, time_step=2, train_begin=0, train_end=50):
    batch_index = []
    data_train = data[train_begin:train_end]
    data_all = data[0:len(data)]

    # RSI 處理
    rsi = ta.RSI(np.array(data_all[:, 3]), timeperiod=RSI_timeperiod)
    rsi_train = rsi[:train_end]
    rsi_test = rsi[train_end + 1:]

    # 標準分數 Z-Score 感覺不是很適用，開高低收量應該有相應規則，應該分別做正規化數據處理(技術指標)，而非標準分數大鍋一炒即可，以後再改。
    normalized_train_data = np.array(
        (data_train-np.mean(data_train, axis=0)) / np.std(data_train, axis=0), dtype=np.float32)

    # 2290 * 5 = 11450 (5個值: OHLCV)
    train_x = np.empty(shape=[11450, time_step, 5], dtype=np.float32)
    train_y = np.empty(shape=[11450], dtype=np.float32)

    # =2290 （一次抓time_step(5)天，連續抓shift_max(5)次，所以最後的界限會減去(10)天資料
    for i in range(len(normalized_train_data) - time_step - shift_max):
        if i % batch_size == 0:
            batch_index.append(i)
        for shift in range(shift_max):   # shift
            # train_x 要連續抓 time_step 天，才能用一張2D圖，表示其型態
            x = normalized_train_data[i + shift:i + time_step + shift, :5]
            train_x[i * time_step + shift] = x          # [11450 * 5 * 5]
            y = normalized_train_data[i + time_step + shift, 5]
            train_y[i * time_step + shift] = y          # [11450]
            # print(i * time_step + shift)
    # batch_index 加上最後一筆
    batch_index.append((len(normalized_train_data)-time_step - shift_max))
    return batch_index, train_x, train_y  # , rsi_test

# 取得訓練資料 2
def get_train_data2(batch_size=10, time_step=2, train_begin=0, train_end=50):
    batch_index = []
    data_train = data[train_begin:train_end]

    # 標準分數 Z-Score 感覺不是很適用，開高低收量應該有相應規則，應該分別做正規化數據處理(技術指標)，而非標準分數大鍋一炒即可，以後再改。
    normalized_train_data = (
        data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)

    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step):  # 一次抓 time_step(5)天資料當做一個訓練單位
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :5]
        y = normalized_train_data[i:i + time_step, 5, np.newaxis]
        train_x.append(x.tolist())      # [2295 * 5 * 5]
        train_y.append(y.tolist())      # [2295 * 5]
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y

# 取得測試資料 1
def get_test_data(time_step=2, test_begin=50):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)

    normalized_test_data = np.array((data_test-mean) / std, dtype=np.float32)

    test_x = np.empty(shape=[len(normalized_test_data) - time_step + 1,
                             time_step, 5], dtype=np.float32)  # 2290 * shift_max = 11450
    test_y = np.empty(
        shape=[len(normalized_test_data) - time_step + 1], dtype=np.float32)

    for i in range(len(normalized_test_data) - time_step + 1):
        x = normalized_test_data[i:i + time_step, :5]
        y = normalized_test_data[i + time_step - 1, 5]
        test_x[i] = x    # [21 * 5 * 5]
        test_y[i] = y    # [21]

    return mean, std, test_x, test_y

# 取得測試資料 2
def get_test_data2(time_step=2, test_begin=50):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)

    normalized_test_data = (data_test-mean) / std

    # test_x2 的 shape 要跟 train_x2 的 time_step 定的 shape 一樣，否則 LSTM 輸入會不一致，無法繼續運算
    size = (len(normalized_test_data)+time_step -
            1)//time_step  # 有 size 個 sample 之意

    test_x, test_y = [], []
    for i in range(size-1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :5]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 5]
        test_x.append(x.tolist())
        test_y.extend(y)
    # 基本上是裝成 [5 * 5 * 5(只有1個元素)] 最後一個沒裝滿，樣本不足
    test_x.append((normalized_test_data[(i + 1) * time_step:, :5]).tolist())
    # 應該跟上列一樣 shape，但裝成 [21] 也能用，因為算 loss2 會 reshape 成一維
    test_y.extend((normalized_test_data[(i + 1) * time_step:, 5]).tolist())
    return mean, std, test_x, test_y

# 神經網路 LSTM 定義
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    # 將 tensor 轉成 2維以進行運算，計算所得結果將作為 hidden layer input
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    # 把 tensor 轉成 3維，用作 LSTM cell input
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])

    # with tf.variable_scope('lstm'):
    cell = tf.contrib.rnn.BasicLSTMCell(
        rnn_unit, forget_bias=0.0, state_is_tuple=True)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_state = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, time_major=False,
                                                dtype=tf.float32)  # output_rnn 記錄 lstm 每個輸出節點的結果，final_states則是最後一個 cell 的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 用作輸出層的輸入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_state

# 開始
plt.figure()

X = tf.placeholder("float", [None, time_step, input_size, 1])
Y = tf.placeholder("float", [None, output_size])
# 每個批次輸入到網路的 tensor
A = tf.placeholder(tf.float32, [None, time_step, input_size])
# 每個批次 tensor 對應的標籤
B = tf.placeholder(tf.float32, [None, time_step, output_size])

train_begin = 0
train_end = 2300  # 共2323筆，取2300筆訓練

# 2330A
batch_index, train_x, train_y = get_train_data(
    batch_size=batch_size, time_step=time_step, train_begin=train_begin, train_end=train_end)
# 因為一次丟 time_step個數據(一組)算，所以要往前再拉 time_Step 天才會跟 RNN 算的天數一致
mean, std, test_x, test_y = get_test_data(
    time_step=time_step, test_begin=2301 - time_step + 1)
batch_index2, train_x2, train_y2 = get_train_data2(
    batch_size=batch_size, time_step=time_step, train_begin=train_begin, train_end=train_end)
mean2, std2, test_x2, test_y2 = get_test_data2(
    time_step=time_step, test_begin=2301)

# CNN 初始化
w1 = init_weights([3, 3, 1, 32])
# w2 = init_weights([32976, 100])   # output 100 給下個 RNN 當作特徵值 input
w2 = init_weights([288, 1])

trX = train_x.reshape(-1, time_step, 5, 1)
trY = train_y.reshape(-1, 1)
teX = test_x.reshape(-1, time_step, 5, 1)
teY = test_y.reshape(-1, 1)

# !!! 不能換置放其他行列位置，要在 teY 指定完畢之後
test_y = np.array(test_y) * std[5] + mean[5]
test_y2 = np.array(test_y2) * std2[5] + mean2[5]

# CNN 呼叫
y_cnn = cnn(X, w1, w2)

# RNN 呼叫
pred, final_state = lstm(A)

# 損失函數定義
loss = tf.reduce_mean(tf.square(tf.reshape(y_cnn, [-1]) - tf.reshape(Y, [-1])))
loss2 = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(B, [-1])))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)
train_op2 = tf.train.AdamOptimizer(lr).minimize(loss2)

with tf.Session() as sess:
    # For showing in TensorBoard
    writer = tf.summary.FileWriter(DataPath, sess.graph)
    sess.run(tf.global_variables_initializer())
    # saver.restore

    # 訓練 500 次 (次數可調整，較高次數可獲得較佳準度)
    for i in range(500):
        for step in range(len(batch_index)-1):
            final_state, loss_ = sess.run([train_op, loss], feed_dict={
                                          X: trX[batch_index[step]:batch_index[step + 1]], Y: trY[batch_index[step]:batch_index[step + 1]]})
            final_state2, loss2_ = sess.run([train_op2, loss2], feed_dict={
                                            A: train_x2[batch_index[step]:batch_index[step + 1]], B: train_y2[batch_index[step]:batch_index[step + 1]]})

            print("i: %d, step: %d, loss_: %f, loss2_: %f" %
                  (i, step, loss_, loss2_))

        test_predict, test_predict2 = [], []
        for step2 in range(len(teX)):
            prob = sess.run(y_cnn, feed_dict={X: [teX[step2]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)

            if step2 < 4:
                prob2 = sess.run(pred, feed_dict={A: [test_x2[step2]]})
                predict2 = prob2.reshape((-1))
                test_predict2.extend(predict2)
                print ("step2: %d" % (step2))

        # 逆標準化，把值還原
        test_predict = np.array(test_predict) * std[5] + mean[5]
        test_predict2 = np.array(test_predict2) * std2[5] + mean2[5]
        acc = np.average(np.abs(
            test_predict - teY[:len(test_predict)]) / teY[:len(test_predict)])  # acc为测试集偏差

        # print("i: %d loss: %f" %(i, loss_, loss2_))
        # plt.scatter(i, loss_)
        '''
        if i % 200==0:
            print("保存模型：", saver.save(sess, 'stock2.model', global_step=i))
        '''

plt.plot(test_y, label='2330 close')
plt.plot(test_predict, label='CNN predict')
plt.plot(test_predict2, label='RNN predict')
plt.legend(loc='upper left')
plt.show()
