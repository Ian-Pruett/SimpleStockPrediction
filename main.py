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

def load_dataset(stock):
    filepath = base_dir + 'data/Stocks/' + stock + '.us.txt'
    data = np.loadtxt(fname = filepath, dtype=str, delimiter=',')
    data = data[1:]
    data = data[:,:2]
    return data


# look_back is the number of times steps to look back
# where as look_ahead is the number of times steps
# to predict
def process_dataset(D,look_back=1,look_ahead=1):
    
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


def split_dataset_train_test(D,test_size=.33):
    t = int(len(D) * (1-test_size))
    D_train = D[0:t]
    D_test = D[t:len(D)]
    return D_train,D_test


# given Sequential model and input X,
# predict y up to k steps in the future
# default is to predict 1 step fowards
def time_series_predict(model,X,k=1):
    
    p = len(X[0]) # previous steps looking back

    def predict_foward_steps(x):
        # queue predictions foward
        queue = np.zeros(k) 

        for i in range(k):
            # allocate array d to be made of
            # values of x and prev predictions
            d = np.zeros(len(x))
            # add in x values to d
            d[0:(p-i)] = x[i:(i+p)]
            # add in previous predicted values
            # ONLY IF previous predictions were made
            if len(np.where(queue != 0)) > 0:
                d[(p-i):(p)] = queue[0:i]
            # make prediction and add to queue
            d = np.expand_dims(d,axis=1)
            d = np.expand_dims(d,axis=2)
            print(d.shape)
            thing = model.predict([d])
            print(thing.shape)
            print(thing)
            queue[i] = model.predict([d])

        return queue

    return np.apply_along_axis(predict_foward_steps,1,X)            


def simple_lstm_network(time_steps):
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
    model.add((Dense(1)))
    

    model.compile(
        loss='mse',
        optimizer='adam',
        # metrics=['accuracy','mae']
        metrics=['accuracy']
    )

    return model


def main():
    #preprocessing the data
    D = load_dataset('aapl')
    D = D[:,1].astype(float)
    D = np.expand_dims(D,axis=2)

    time_steps = 30
    predict_steps = 7

    D_test,D_train = split_dataset_train_test(D)

    X_train,y_train = process_dataset(
        D_train,
        look_back=time_steps,
        look_ahead=1
    )

    X_test,y_test = process_dataset(
        D_test,
        look_back=time_steps,
        look_ahead=predict_steps
    )

    X_train = np.expand_dims(X_train,axis=2)
    X_test = np.expand_dims(X_test,axis=2)
    # y = np.expand_dims(y,axis=1)

    # X_train,X_test,y_train,y_test = split_train_test(
    #     X,y,
    #     test_size=.33
    # )

    model = simple_lstm_network(
        time_steps=time_steps
    )

    model.fit(
        X_train,y_train,
        batch_size=10,
        epochs=5
    )

    # predictions = model.predict([X_test])
    predictions = time_series_predict(
        model,X_train,
        k=predict_steps
    )

    # loss, acc, mae = model.evaluate(X_test,y_test)
    # print('Loss: %f\tAcc: %f\tMAE: %f' % (loss,acc,mae))
    loss, acc = model.evaluate(X_test,y_test)
    print('Loss: %f\tAcc: %f' % (loss,acc))

    plt.plot(y_test.flatten())
    plt.plot(predictions.flatten())
    plt.show()
   

if __name__ == '__main__':
    main()
