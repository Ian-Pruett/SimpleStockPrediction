import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.utils import plot_model


# define base directory this changes 
# when running in Google CoLab 
base_dir = ''


def load_dataset(stock):
    filepath = base_dir + 'data/Stocks/' + stock + '.us.txt'
    data = np.loadtxt(fname = filepath, dtype=str, delimiter=',')
    data = data[1:]
    data = data[:,:2]
    return data


# indexes time-series data into features and 
# corresponding output. For each vector x in X
# there is a corresponding y
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
        if len(X_entry) != look_back:   # alternative to fixing
            break                       # index errors, got lazy . . .
        if len(y_entry) != look_ahead:
            break
        X.append(X_entry)
        y.append(y_entry)

    return np.array(X),np.array(y)


# Splits dataset into train and test given 
# given features X, output y, and the 
# size of test sample
def split_train_test(X,y,test_size=.33):
    t = int(len(X) * (1-test_size))
    X_train = X[0:t]
    X_test = X[t:len(X)]
    y_train = y[0:t]
    y_test = y[t:len(y)]
    return X_train,X_test,y_train,y_test


# Splits dataset into train and test given 
# given original dataset size of test sample
def split_dataset_train_test(D,test_size=.33):
    t = int(len(D) * (1-test_size))
    D_train = D[0:t]
    D_test = D[t:len(D)]
    return D_test,D_train


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
            d = np.array([d])
            queue[i] = model.predict([d]).flatten()

        return queue # set of predictions of k steps

    return np.apply_along_axis(predict_foward_steps,1,X)            


def simple_lstm_network(time_steps, N):
    model = Sequential()

    # first layer  
    model.add(LSTM(
        units=time_steps,
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        unroll=False,
        use_bias=True,
        name='Input_LSTM'
    ))
    
    model.add(Dropout(0.2))

    # second layer
    model.add(LSTM(
        units= N // time_steps,
        # return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        unroll=False,
        use_bias=True,
        name='Hidden_LSTM'
    ))
    
    model.add(Dropout(0.2))

    # output layer
    model.add((Dense(1,name='Output')))

    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['accuracy','mae']
    )

    return model


# return the error over k steps foward
# error is returned as numpy array and
# given by whatever metric is assigned
def err_over_steps(D,p,k,model,metric):
    err = np.zeros(k)  # allocate array
    # predict time steps from 1 to k steps
    for i in range(1,k + 1):
        X,y_true = process_dataset(
            D,look_back=p,look_ahead=i
        )
        y_pred = time_series_predict(
            model,X,k=i
        )

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        err[i - 1] = metric(y_true,y_pred)
        print(err[i - 1])

    return err


def main(stock):

    #preprocessing the data
    D = load_dataset(stock)
    D = D[:,1].astype(float)
    D = np.expand_dims(D,axis=2)

    # define time steps, prev and foward
    time_steps = 30
    predict_steps = 20

    D_test,D_train = split_dataset_train_test(D)
    N = len(D_train)
    testHigh = np.max(D_test)
    testLow = np.min(D_test)
    testDelta = testHigh - testLow

    # index training and test sets
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

    # reshape to train lstm network
    X_train = np.expand_dims(X_train,axis=2)
    X_test = np.expand_dims(X_test,axis=2)

    model = simple_lstm_network(
        time_steps=time_steps,
        N=N
    )

    model.fit(
        X_train,y_train,
        batch_size=10,
        epochs=5
    )

    fname = 'visuals/model_' + stock + '.png'
    plot_model(model, to_file=fname, show_shapes=True)

    y_pred = time_series_predict(
        model,X_test,k=predict_steps
    )

    y_test = y_test.flatten()
    y_pred = y_pred.flatten()

    y_Pred = testDelta * y_pred + testLow
    y_Test = testDelta * y_test + testLow
    plt.plot(y_Test)
    plt.plot(y_Pred)
    plt.xlabel('time steps (days)')
    plt.ylabel('price ($)')
    plt.title('Opening Price Over Time (%s)' % stock)
    fname = 'visuals/complete_network_' + stock + '.png'
    plt.savefig(fname)
    plt.clf()

    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)

    # print('MSE: %f\tMAE: %f' % (mse,mae))
    
    # error over predict_steps
    err = err_over_steps(
        D_test,
        time_steps,
        predict_steps,
        model,
        mean_absolute_error
    )

    plt.xlabel('foward time steps')
    plt.ylabel('mae')
    plt.title('Mean Absolute Error Over Predicted Steps (%s)' % stock)
    plt.plot(err)
    fname = 'visuals/error_over_steps' + stock + '.png'
    plt.savefig(fname)
    plt.clf()
   

if __name__ == '__main__':
    stocks = ['aapl','fb','msft','ibm','ba','nflx','amzn','aac','tsla','twtr','bp']
    for stock in stocks:
        main(stock)