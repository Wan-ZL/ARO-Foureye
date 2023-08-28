import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import pickle
import os
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
# import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def generate_trained_ML(schemes):
    for schemes_index in range(len(schemes)):
        all_dataset_X = np.zeros((1, strategy_number))
        all_dataset_Y = np.zeros((1, strategy_number))
        path = "data/trainning_data/" + schemes[schemes_index]
        file_list = [f for f in os.listdir(path) if not f.startswith('.')]
        if len(file_list) == 0:
            print("!! "+schemes[schemes_index]+" No File")
            continue
        for file_name in file_list:
            print("data/trainning_data/" + schemes[schemes_index] + "/"+file_name)
            the_file = open("data/trainning_data/" + schemes[schemes_index] + "/"+file_name, "rb")
            all_result_after_each_game_all_result = pickle.load(the_file)
            the_file.close()


            for key in all_result_after_each_game_all_result.keys():
                # transfer c to S
                # S = np.array(np.sum(all_result_after_each_game_all_result[key], axis=1))
                S = np.array(all_result_after_each_game_all_result[key])

                # add [0....0] to head
                S_with_zero_head = np.concatenate((np.zeros((1,strategy_number)), S), axis=0)

                # concatenate to dataset
                all_dataset_X = np.concatenate((all_dataset_X, S_with_zero_head[:-1]), axis=0)
                all_dataset_Y = np.concatenate((all_dataset_Y, S_with_zero_head[1:]), axis=0)

        # # !!!================ below estimate each strategy seperately ================
        # all_dataset_X_normalized = array_normalization(all_dataset_X)
        # all_dataset_Y_normalized = array_normalization(all_dataset_Y)
        #
        # model_list = []
        # for index in range(9):
        #     # strate_dataset_X = all_dataset_X_normalized[:,index].reshape(-1, 1)
        #     # strate_dataset_Y = all_dataset_Y_normalized[:, index]
        #     strate_dataset_X = all_dataset_X_normalized[:, index]
        #     strate_dataset_Y = all_dataset_Y_normalized[:, index]
        #
        #     section_size = 3
        #     strate_dataset_section_X = np.array([[[strate_dataset_X[i+j]] for i in range(section_size)] for j in range(strate_dataset_X.shape[0]-section_size)])
        #     strate_dataset_section_X = np.array([[strate_dataset_X[i + j] for i in range(section_size)] for j in
        #                                          range(strate_dataset_X.shape[0] - section_size)])
        #     strate_dataset_section_Y = strate_dataset_Y[section_size:]
        #
        #
        #     X_train, X_test, y_train, y_test = train_test_split(strate_dataset_section_X, strate_dataset_section_Y, test_size=0.1, random_state=1)
        #
        #     # KNN
        #     n_neighbors = 5
        #     model = neighbors.KNeighborsRegressor(n_neighbors, weights='distance', algorithm='brute')\
        #         .fit(X_train, y_train)
        #     model_list.append(model)
        #
        #     # LSTM
        #     # model = Sequential()
        #     # model.add(LSTM((1), batch_input_shape=(None,section_size,1), return_sequences=False))
        #     # model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
        #     # model.summary()
        #     # history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)
        #
        #
        #     # print(f"predict {model.score(X_test, y_test)}")
        #     y_predict = model.predict(X_test)
        #     # print(X_test[:,-1])
        #     print(f"predict {r2_score(y_test, y_predict)}")
        #     print(f"no predict {r2_score(y_test, X_test[:,-1])}")
        #
        #
        #
        #
        #
        # # print(all_dataset_X.shape[0])
        # # section_size = 3
        # # section_dataset_X = np.array([[all_dataset_X[i+j] for i in range(section_size)] for j in range(all_dataset_X.shape[0]-section_size)])
        # # print(section_dataset_X)
        # # print(section_dataset_X.shape)
        # # print(all_dataset_Y.shape)
        # # print(all_dataset_Y[section_size:].shape)
        # # n_neighbors = 5
        # # knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance', algorithm='brute')
        # # model = knn.fit(section_dataset_X, all_dataset_Y[section_size:])
        # print("return")
        # return
        # # ================ above estimate each strategy separately ================
        print(all_dataset_X.shape)
        all_dataset_X_normalized = array_normalization(all_dataset_X)
        all_dataset_Y_normalized = array_normalization(all_dataset_Y)

        X_train, X_test, y_train, y_test = train_test_split(all_dataset_X_normalized, all_dataset_Y_normalized, test_size=0.1, random_state=1)
        # #
        n_neighbors = 50
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance', algorithm='brute')
        model = knn.fit(X_train, y_train)

        # for test
        # model = MultiOutputRegressor(GradientBoostingRegressor(random_state=0), n_jobs=-1).fit(X_train, y_train)
        # model = MLPRegressor(hidden_layer_sizes=(100, 100), random_state=1, max_iter=1000).fit(X_train, y_train)
        # model = LinearRegression().fit(X_train, y_train)



        y_predict = model.predict(X_test)
        (row,column) = X_test.shape
        y_random = np.random.rand(row,column)

        # y_predict_base = np.sum(y_predict, axis=1)
        # y_predict_normalized = array_normalization(y_predict)
        # y_test_normalized = array_normalization(y_test)
        # X_test_normalized = array_normalization(X_test)

        # print(array_normalization(X_test))
        # print(model.score(X_test, y_test))

        print("\nbuildin score:")
        print(f"predict {model.score(X_test, y_test)}")

        print("\nR Square:")
        print(f"predict {r2_score(y_test, y_predict)}")
        print(f"no predict {r2_score(y_test, X_test)}")
        print(f"random {r2_score(y_test, y_random)}")

        print("\nMean Square Error")
        print(f"predict {mean_squared_error(y_test, y_predict)}")
        print(f"no predict {mean_squared_error(y_test, X_test)}")
        print(f"random {mean_squared_error(y_test, y_random)}")

        print("\nMean Absolute Error")
        print(f"predict {mean_absolute_error(y_test, y_predict)}")
        print(f"no predict {mean_absolute_error(y_test, X_test)}")
        print(f"random {mean_absolute_error(y_test, y_random)}")



        # save trained model
        os.makedirs("data/trained_ML_model", exist_ok=True)
        the_file = open("data/trained_ML_model/knn_trained_model_"+schemes[schemes_index]+".pkl", "wb+")
        pickle.dump(model, the_file)
        the_file.close()

def array_normalization(_2d_array):
    sum_array = np.ones((len(_2d_array),strategy_number))/strategy_number
    for index in range(len(_2d_array)):
        if sum(_2d_array[index]) == 0:
            continue
        else:
            sum_array[index] = _2d_array[index]/sum(_2d_array[index])

    # for array in _2d_array:
    #     if sum(array) == 0:
    #         sum_array = np.append(sum_array, 0)
    #     else:
    #         sum_array = np.append(sum_array, sum(array))
    return sum_array



def display_prediction_result(schemes):
    figure_high = 6
    figure_width = 7.5

    for schemes_index in range(len(schemes)):
        print(schemes[schemes_index])
        all_dataset_X = np.zeros((1, strategy_number))
        all_dataset_Y = np.zeros((1, strategy_number))
        path = "data/trainning_data/" + schemes[schemes_index]
        file_list = [f for f in os.listdir(path) if not f.startswith('.')]
        if len(file_list) == 0:
            print("!! " + schemes[schemes_index] + " No File")
            continue
        file_name = file_list[0]
        the_file = open("data/trainning_data/" + schemes[schemes_index] + "/" + file_name, "rb")
        all_result_after_each_game_all_result = pickle.load(the_file)
        the_file.close()

        the_file = open("data/trained_ML_model/knn_trained_model_"+schemes[schemes_index]+".pkl", "rb")
        regression_model = pickle.load(the_file)

        part_data = []
        part_data_predict = []
        iteration_index = 0
        array_index = 0
        S = array_normalization(all_result_after_each_game_all_result[iteration_index])
        S_pred = regression_model.predict(S)
        # print(S_pred)
        for S_array in S:
            part_data.append(S_array[array_index])

        for S_array in S_pred:
            part_data_predict.append(S_array[array_index])
        print(part_data)
        print(part_data_predict)

        plt.figure(figsize=(figure_width, figure_high))
        plt.plot(range(len(part_data)), part_data, label="Original Data")
        plt.plot(range(1,len(part_data)+1), part_data, label="No Predict Data")
        plt.plot(range(len(part_data)), part_data_predict, label="Predict Data")
        plt.legend()
        plt.show()




if __name__ == '__main__':
    # schemes = ["DD-IPI", "DD-ML-IPI", "DD-Random-IPI"]
    schemes = ["DD-PI", "DD-IPI"]
    strategy_number = 8
    generate_trained_ML(schemes)
    # display_prediction_result(schemes)







