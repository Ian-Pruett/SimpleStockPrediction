import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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


def process_dataset(frame, look_back=1):
    D = frame.values.astype('float32')
    
    # scale data between 0,1
    scaler = MinMaxScaler(feature_range=(0,1))
    D = scaler.fit_transform(D)
    
    X,y = [],[]
    
    for i in range(len(D)-look_back-1):
        entry = D[i:(i+look_back), 0]
        X.append(entry)
        y.append(D[i + look_back, 0])
    
    return np.array(X),np.array(y)


def simple_lstm_network():
    model = keras.Sequential()

    # first layer
    model.add(keras.layers.LSTM(
        units=30,
        activation='tanh',
        recurrent_activation='sigmoid',
        recurrent_dropout=0,
        unroll=False,
        use_bias=True,
    ))
    
    model.add(keras.layers.Dropout(0.2))

    # output layer
    model.add(keras.layers.Dense(1))
    

    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['accuracy','mae']
    )

    return model


def main():
    dataframe = load_dataframe('aapl')

    # plt.plot(dataframe)
    # plt.show()

    X,y = process_dataset(dataframe,look_back=30)

    X = np.expand_dims(X,axis=2)
    y = np.expand_dims(y,axis=1)

    model = simple_lstm_network()

    X_train = X[:6664]
    y_train = y[:6664]

    X_test = X[6664:]
    y_test = y[6664:]

    model.fit(
        X_train,y_train,
        batch_size=10,
        epochs=5
    )

    predictions = model.predict([X_test])

    loss, acc, mae = model.evaluate(X_test,y_test)
    print('Loss: %f\tAcc: %f\tMAE: %f' % (loss,acc,mae))

    plt.plot(y_test)
    plt.plot(predictions)
    # plt.show()

    # lets try predicting a year with only a month

    past = np.copy(X_test)
    generated = []
    days = len(X_test)

    for d in range(1,days-1):
        
        new_data = np.ones((30,1))
        new_data[0:29] = past[d-1][1:30]
        
        predict = model.predict([[past[d-1]]])
        new_data[29:30] = np.array(predict)
        generated.append(predict)
        
        past[d] = new_data

    generated = np.array(generated)

    generated = generated.reshape(-1, generated.shape[-1])

    print(y_test.shape)
    print(generated.shape)

    plt.plot(generated)
    plt.show()

if __name__ == '__main__':
    main()