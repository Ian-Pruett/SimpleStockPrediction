import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, TimeDistributed


# define base directory
base_dir = ''


def load_dataframe(stock):
    # read file into dataframe
    filepath = base_dir + 'data/processed/'+ stock + '.open.txt'
    dataframe = pd.read_csv(
        filepath,
        usecols=[1], 
        engine='python',
        skipfooter=3
    )
    return dataframe


# look_back is the number of times steps to look back
# where as look_ahead is the number of times steps
# to predict
def process_dataset(frame,look_back=1,look_ahead=1):
    D = frame.values.astype('float32')
    
    # scale data between 0,1
    scaler = MinMaxScaler(feature_range=(0,1))
    D = scaler.fit_transform(D)
    
    X,y = [],[]
    
    i = 0
    n = len(D)-look_back-1

    for i in range(0,n,look_ahead):
        j = i + look_back
        k = j + look_ahead
        X_entry = D[i:j,0]
        y_entry = D[j:k,0]
        X.append(X_entry)
        y.append(y_entry)

    return np.array(X),np.array(y)


def split_train_test(X,y,test_size=.33):
    t = int(len(X) * (1-test_size))
    X_train = X[0:t]
    X_test = X[t:len(X)]
    y_train = y[0:t]
    y_test = y[t:len(y)]
    return X_train,X_test,y_train,y_test


def simple_lstm_network(time_steps,predict_steps):
    model = Sequential()

    # first layer
    model.add(LSTM(
        units=time_steps,
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        unroll=False,
        use_bias=True,
    ))
    
    model.add(Dropout(0.2))

    # second layer
    model.add(LSTM(
        units=time_steps // 2,
        # return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        unroll=False,
        use_bias=True,
    ))
    
    model.add(Dropout(0.2))

    # output layer
    # model.add(TimeDistributed(Dense(predict_steps)))
    model.add((Dense(predict_steps)))
    

    model.compile(
        loss='mse',
        optimizer='adam',
        # metrics=['accuracy','mae']
        metrics=['accuracy']
    )

    return model


def main():
    dataframe = load_dataframe('aapl')

    time_steps = 30
    predict_steps = 1

    X,y = process_dataset(
        dataframe,
        look_back=time_steps,
        look_ahead=predict_steps
    )

    X = np.expand_dims(X,axis=2)
    # y = np.expand_dims(y,axis=1)

    X_train,X_test,y_train,y_test = split_train_test(
        X,y,
        test_size=.33
    )

    model = simple_lstm_network(
        time_steps=time_steps,
        predict_steps=predict_steps,
    )

    model.fit(
        X_train,y_train,
        batch_size=10,
        epochs=5
    )

    predictions = model.predict([X_test])

    # loss, acc, mae = model.evaluate(X_test,y_test)
    # print('Loss: %f\tAcc: %f\tMAE: %f' % (loss,acc,mae))
    loss, acc = model.evaluate(X_test,y_test)
    print('Loss: %f\tAcc: %f' % (loss,acc))

    plt.plot(y_test.flatten())
    plt.plot(predictions.flatten())
    plt.show()
   

if __name__ == '__main__':
    main()