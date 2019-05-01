from datetime import date
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

# last update to dataset 11/10/2019
t0 = date(2019,11,10)

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
#        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        unroll=False,
        use_bias=True,
        name='Input_LSTM'
    ))
    
    model.add(Dropout(0.2))

    # second layer
#    model.add(LSTM(
#        units= N // time_steps,
#        # return_sequences=True,
#        activation='tanh',
#        recurrent_activation='sigmoid',
#        unroll=False,
#        use_bias=True,
#        name='Hidden_LSTM'
#    ))
    
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

    #enter time delta
    t = date(2007, 11, 10)
    delta = (t0-t).days
    D = D[(D.size-delta):]

    D = np.expand_dims(D,axis=2)

    # define time steps, prev and foward
    time_steps = 30
    predict_steps = 10

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

    fname = base_dir + 'visuals/model_' + stock + '.png'
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
    fname = base_dir + 'visuals/complete_network_' + stock + '.png'
    plt.savefig(fname)
    plt.clf()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print('MSE: %f\tMAE: %f' % (mse,mae))
    
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


def main2(stocks):
    stock1 = stocks[0]
    stock2 = stocks[1]
    stock3 = stocks[2]

    #preprocessing the data
    D1 = load_dataset(stock1)
    D1 = D1[:,1].astype(float)

    #enter time delta
    t = date(2007, 11, 10)
    delta = (t0-t).days
    D1 = D1[(D1.size-delta):]

    D1 = np.expand_dims(D1,axis=2)

    #preprocessing the data
    D2 = load_dataset(stock2)
    D2 = D2[:,1].astype(float)

    #enter time delta
    t = date(2007, 11, 10)
    delta = (t0-t).days
    D2 = D2[(D2.size-delta):]

    D2 = np.expand_dims(D2,axis=2)

    #preprocessing the data
    D3 = load_dataset(stock3)
    D3 = D3[:,1].astype(float)

    #enter time delta
    t = date(2007, 11, 10)
    delta = (t0-t).days
    D3 = D3[(D3.size-delta):]

    D3 = np.expand_dims(D3,axis=2)

    # define time steps, prev and foward
    time_steps = 30
    predict_steps = 10

    D1_test,D1_train = split_dataset_train_test(D1)
    N1 = len(D1_train)

    D2_test,D2_train = split_dataset_train_test(D2)
    testHigh2 = np.max(D2_test)
    testLow2 = np.min(D2_test)
    testDelta2 = testHigh2 - testLow2

    D3_test,D3_train = split_dataset_train_test(D3)
    testHigh3 = np.max(D3_test)
    testLow3 = np.min(D3_test)
    testDelta3 = testHigh3 - testLow3

    # index training and test sets
    X1_train,y1_train = process_dataset(
        D1_train,
        look_back=time_steps,
        look_ahead=1
    )

    X2_test,y2_test = process_dataset(
        D2_test,
        look_back=time_steps,
        look_ahead=predict_steps
    )

    X3_test,y3_test = process_dataset(
        D3_test,
        look_back=time_steps,
        look_ahead=predict_steps
    )

    # reshape to train lstm network
    X1_train = np.expand_dims(X1_train,axis=2)
    X2_test = np.expand_dims(X2_test,axis=2)
    X3_test = np.expand_dims(X3_test,axis=2)

    model = simple_lstm_network(
        time_steps=time_steps,
        N=N1
    )

    model.fit(
        X1_train,y1_train,
        batch_size=10,
        epochs=5
    )

    y2_pred = time_series_predict(
        model,X2_test,k=predict_steps
    )

    y2_test = y2_test.flatten()
    y2_pred = y2_pred.flatten()

    y2_Pred = testDelta2 * y2_pred + testLow2
    y2_Test = testDelta2 * y2_test + testLow2
    plt.plot(y2_Test)
    plt.plot(y2_Pred)
    plt.xlabel('time steps (days)')
    plt.ylabel('price ($)')
    plt.title('Opening Price Over Time (%s)' % stock2)
    plt.show()

    mae = mean_absolute_error(y2_test, y2_pred)
    mse = mean_squared_error(y2_test, y2_pred)

    print('MSE: %f\tMAE: %f' % (mse,mae))

    y3_pred = time_series_predict(
        model,X3_test,k=predict_steps
    )

    y3_test = y3_test.flatten()
    y3_pred = y3_pred.flatten()

    y3_Pred = testDelta3 * y3_pred + testLow3
    y3_Test = testDelta3 * y3_test + testLow3
    plt.plot(y3_Test)
    plt.plot(y3_Pred)
    plt.xlabel('time steps (days)')
    plt.ylabel('price ($)')
    plt.title('Opening Price Over Time (%s)' % stock3)
    plt.show()

    mae = mean_absolute_error(y3_test, y3_pred)
    mse = mean_squared_error(y3_test, y3_pred)

    print('MSE: %f\tMAE: %f' % (mse,mae))
   

if __name__ == '__main__':
    # stocks = ['aapl','fb','msft','ibm','ba','nflx',
    #         'amzn','aac','tsla','twtr','bp']
    # for stock in stocks:
    #     main(stock)
    stocks =  ['aapl','ge','ppg']
    main2(stocks)