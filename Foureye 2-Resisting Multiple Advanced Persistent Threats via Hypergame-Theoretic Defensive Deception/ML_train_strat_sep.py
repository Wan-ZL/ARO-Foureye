import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import pickle
import os
import collections
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM


def train_ML_Belief_to_Belief(schemes, window_size, n_neighbors, strategy_number):
    for schemes_index in range(len(schemes)):
        all_dataset_X = np.zeros((1, strategy_number))
        all_dataset_Y = np.zeros((1, strategy_number))
        original_belief = np.zeros((1, strategy_number))
        path = "data/trainning_data/" + schemes[schemes_index]
        file_list = [f for f in os.listdir(path) if not f.startswith('.')]
        if len(file_list) == 0:
            print("!! "+schemes[schemes_index]+" No File")
            continue
        for file_name in file_list:
            print("data/trainning_data/" + schemes[schemes_index] + "/"+file_name)
            the_file = open("data/trainning_data/" + schemes[schemes_index] + "/"+file_name, "rb")
            all_result_def_belief_all_result = pickle.load(the_file)
            the_file.close()


            for key in all_result_def_belief_all_result.keys():
                # transfer c to S
                # S = np.array(np.sum(all_result_after_each_game_all_result[key], axis=1))
                S = np.array(all_result_def_belief_all_result[key])

                # padding: [0....0]*window_size to head
                S_with_zero_head = np.concatenate((np.zeros((window_size,strategy_number)), S), axis=0)

                # concatenate to dataset
                all_dataset_X = np.concatenate((all_dataset_X, S_with_zero_head[:-1]), axis=0)
                all_dataset_Y = np.concatenate((all_dataset_Y, S_with_zero_head[1:]), axis=0)
                original_belief = np.concatenate((original_belief, S_with_zero_head[:-1]), axis=0)


        all_dataset_X_normalized = array_normalization(all_dataset_X)
        all_dataset_Y_normalized = array_normalization(all_dataset_Y)
        original_belief_normalized = array_normalization(original_belief)

        model_list = []
        total_R2_predict = 0
        total_R2_no_predict = 0
        total_MSE_predict = 0
        total_MSE_no_predict = 0
        for index in range(strategy_number):
            # strate_dataset_X = all_dataset_X_normalized[:,index].reshape(-1, 1)
            # strate_dataset_Y = all_dataset_Y_normalized[:, index]
            strate_dataset_X = all_dataset_X_normalized[:, index]
            strate_dataset_Y = all_dataset_Y_normalized[:, index]
            strate_origin_belief = original_belief_normalized[:, index]


            # window_size = 5
            # strate_dataset_section_X = np.array([[[strate_dataset_X[i+j]] for i in range(window_size)] for j in range(strate_dataset_X.shape[0]-window_size)])
            strate_dataset_section_X = np.array([[strate_dataset_X[i + j] for i in range(window_size)] for j in
                                                 range(strate_dataset_X.shape[0] - window_size)])


            strate_dataset_section_Y = strate_dataset_Y[window_size:]
            strate_origin_belief_section = strate_origin_belief[window_size:]

            pd_strate_dataset_section_Y = pd.DataFrame(strate_dataset_section_Y)
            pd_strate_origin_belief_section = pd.DataFrame(strate_origin_belief_section)


            X_train, X_test, y_train, y_test = train_test_split(strate_dataset_section_X, pd_strate_dataset_section_Y, test_size=0.1, random_state=1)

            # KNN
            model = neighbors.KNeighborsRegressor(n_neighbors, weights='distance', algorithm='brute', n_jobs=-1).fit(X_train, y_train)
            # model = svm.SVR().fit(X_train, y_train)
            model_list.append(model)



            # LSTM
            # model = Sequential()
            # model.add(LSTM((1), batch_input_shape=(None,window_size,1), return_sequences=False))
            # model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
            # model.summary()
            # history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)


            y_predict = model.predict(X_test)
            total_R2_predict += r2_score(y_test, y_predict)
            total_R2_no_predict += r2_score(y_test, pd_strate_origin_belief_section.iloc[y_test.index])

            total_MSE_predict += mean_squared_error(y_test, y_predict)
            total_MSE_no_predict += mean_squared_error(y_test, pd_strate_origin_belief_section.iloc[y_test.index])

        print(strate_dataset_section_X.shape)
        print("\n total R2 score")
        print(f"predict {total_R2_predict}")
        print(f"no predict {total_R2_no_predict}")
        print("total MSE")
        print(f"predict {total_MSE_predict}")
        print(f"predict {total_MSE_no_predict}")





        # print(all_dataset_X.shape[0])
        # window_size = 3
        # section_dataset_X = np.array([[all_dataset_X[i+j] for i in range(window_size)] for j in range(all_dataset_X.shape[0]-window_size)])
        # print(section_dataset_X)
        # print(section_dataset_X.shape)
        # print(all_dataset_Y.shape)
        # print(all_dataset_Y[window_size:].shape)
        # n_neighbors = 5
        # knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance', algorithm='brute')
        # model = knn.fit(section_dataset_X, all_dataset_Y[window_size:])
        # ================ above estimate each strategy separately ================




        # save trained model
        os.makedirs("data/trained_ML_model_list", exist_ok=True)
        the_file = open("data/trained_ML_model_list/knn_trained_model_"+schemes[schemes_index]+".pkl", "wb+")
        pickle.dump(model_list, the_file)
        the_file.close()


def train_ML_Action_to_Belief(schemes, window_size, n_neighbors, strategy_number):
    for schemes_index in range(len(schemes)):
        all_dataset_X = np.zeros((1, window_size, strategy_number))
        all_dataset_Y = np.zeros((1, strategy_number))
        original_belief = np.zeros((1, strategy_number))
        path = "data/trainning_data/" + schemes[schemes_index]
        file_list = [f for f in os.listdir(path) if not f.startswith('.')]
        if len(file_list) == 0:
            print("!! "+schemes[schemes_index]+" No File")
            continue

        # for each file
        for file_name in file_list:
            print("data/trainning_data/" + schemes[schemes_index] + "/"+file_name)
            the_file = open("data/trainning_data/" + schemes[schemes_index] + "/"+file_name, "rb")
            all_result_def_obs_action_all_result, all_result_def_belief_all_result = pickle.load(the_file)
            the_file.close()

            # concatenate data
            # for each simulation
            for key in all_result_def_obs_action_all_result.keys():
                window_x = np.zeros((window_size, strategy_number))
                # for each game
                for record in all_result_def_obs_action_all_result[key]:
                    # update the window
                    window_x = np.vstack([window_x, record])
                    window_x = np.delete(window_x, 0, 0)
                    all_dataset_X = np.vstack((all_dataset_X, [window_x]))

                belief = np.array(all_result_def_belief_all_result[key])

                # Aligning data
                all_dataset_Y = np.concatenate((all_dataset_Y, belief[1:]), axis=0)
                original_belief = np.concatenate((original_belief, belief[:-1]), axis=0)
                all_dataset_X = all_dataset_X[:-1]


        model_list = []
        total_R2_predict = 0
        total_R2_no_predict = 0
        total_MSE_predict = 0
        total_MSE_no_predict = 0
        for index in range(strategy_number):
            strate_dataset_X = all_dataset_X[:, :, index]
            strate_dataset_Y = all_dataset_Y[:, index]
            strate_origin_belief = original_belief[:, index]
            pd_strate_dataset_Y = pd.DataFrame(strate_dataset_Y)
            pd_strate_origin_belief = pd.DataFrame(strate_origin_belief)

            X_train, X_test, y_train, y_test = train_test_split(strate_dataset_X, pd_strate_dataset_Y, test_size=0.1, random_state=1)

            # KNN
            model = neighbors.KNeighborsRegressor(n_neighbors, weights='distance', algorithm='brute', n_jobs=-1).fit(X_train, y_train)
            model_list.append(model)

            y_predict = model.predict(X_test)

            total_R2_predict += r2_score(y_test, y_predict)
            total_R2_no_predict += r2_score(y_test, pd_strate_origin_belief.iloc[y_test.index])
            total_MSE_predict += mean_squared_error(y_test, y_predict)
            total_MSE_no_predict += mean_squared_error(y_test, pd_strate_origin_belief.iloc[y_test.index])

        print(strate_dataset_X.shape)
        print("\n total R2 score")
        print(f"predict {total_R2_predict}")
        print(f"no predict {total_R2_no_predict}")
        print("total MSE")
        print(f"predict {total_MSE_predict}")
        print(f"predict {total_MSE_no_predict}")

        # save trained model
        os.makedirs("data/trained_ML_model_list", exist_ok=True)
        the_file = open("data/trained_ML_model_list/knn_trained_model_"+schemes[schemes_index]+".pkl", "wb+")
        pickle.dump(model_list, the_file)
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

def display_prediction_result(schemes, strategy_number):
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

        the_file = open("data/trained_ML_model_LIST/knn_trained_model_"+schemes[schemes_index]+".pkl", "rb")
        regression_model_list = pickle.load(the_file)

        part_data = []
        part_data_predict = []
        iteration_index = 0
        array_index = 0
        S = array_normalization(all_result_after_each_game_all_result[iteration_index])
        strate_dataset_X = S[:,array_index]

        window_size = 5
        strate_dataset_X_paded = np.insert(strate_dataset_X, 0, np.zeros(window_size))

        strate_dataset_window_X = np.array([[strate_dataset_X_paded[i + j] for i in range(window_size)] for j in
                                             range(strate_dataset_X_paded.shape[0] - window_size)])

        part_data_predict = regression_model_list[array_index].predict(strate_dataset_window_X)
        # print(S_pred)

        for S_array in S:
            part_data.append(S_array[array_index])

        # for S_array in S_pred:
        #     part_data_predict.append(S_array[array_index])
        # print(part_data)
        # print(part_data_predict)

        plt.figure(figsize=(figure_width, figure_high))
        plt.plot(range(len(part_data)), part_data, label="Original Data")
        plt.plot(range(1,len(part_data)+1), part_data, label="No Predict Data")
        plt.plot(range(len(part_data)), part_data_predict, label="Predict Data")
        plt.legend()
        plt.show()

def train_ML_predict_action(schemes, x_length, n_neighbors, strategy_number):
    for schemes_index in range(len(schemes)):
        all_dataset_X = []
        all_dataset_Y = []
        path = "data/trainning_data/" + schemes[schemes_index]
        file_list = [f for f in os.listdir(path) if not f.startswith('.')]
        if len(file_list) == 0:
            print("!! "+schemes[schemes_index]+" No File")
            continue
        for file_name in file_list:
            print("data/trainning_data/" + schemes[schemes_index] + "/"+file_name)
            the_file = open("data/trainning_data/" + schemes[schemes_index] + "/"+file_name, "rb")
            [ML_x_data_all_result, ML_y_data_all_result] = pickle.load(the_file)
            the_file.close()


            for key in ML_x_data_all_result.keys():
                for dataset_x in ML_x_data_all_result[key]:
                    all_dataset_X.append(dataset_x)
                for dataset_y in ML_y_data_all_result[key]:
                    all_dataset_Y.append(dataset_y)

        # print(np.sum(np.array(all_dataset_Y) == 0)/len(all_dataset_Y))
        # print(np.sum(np.array(all_dataset_Y) == 5) / len(all_dataset_Y))
        # print(np.sum(np.array(all_dataset_Y) == 0) / len(all_dataset_Y) + np.sum(np.array(all_dataset_Y) == 5) / len(all_dataset_Y))
        # print("the X")
        # print(np.sum(np.array(all_dataset_X)[:,24:32], axis=1).shape)
        # print(np.sum(np.array(all_dataset_X)[:,24:32] == 0))
        # print("zero condition:")
        # print(np.sum(np.sum(np.array(all_dataset_X)[:,24:32], axis=1) == 0) / np.sum(np.array(all_dataset_X)[:,24:32], axis=1).size)
        # print(np.mean(np.array(all_dataset_X)[:, 33:41]))
        #
        # print(np.array(all_dataset_X).shape)
        # print(np.array(all_dataset_Y).shape)

        X_train, X_test, y_train, y_test = train_test_split(all_dataset_X, all_dataset_Y,
                                                            test_size=0.1, random_state=1)

        # model = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', algorithm='brute', n_jobs=-1).fit(X_train, y_train)
        # model = neighbors.KNeighborsClassifier().fit(X_train, y_train)
        model = tree.DecisionTreeClassifier().fit(X_train, y_train)

        y_predict = model.predict(X_test)
        R2_predict = r2_score(y_test, y_predict)
        print(f"R2_predict {R2_predict}")
        MSE_predict = mean_squared_error(y_test, y_predict)
        print(f"MSE_predict {MSE_predict}")

        print(model.get_params())
        print(model.tree_.node_count)

        # save trained model
        # os.makedirs("data/trained_ML_model", exist_ok=True)
        # the_file = open("data/trained_ML_model/trained_classi_model_"+schemes[schemes_index]+".pkl", "wb+")
        # pickle.dump(model, the_file)
        # the_file.close()

def train_ML_predict_action_vary_AAP(schemes, x_length, n_neighbors, strategy_number):
    AAP_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for AAP in AAP_list:
        print(f"AAP={AAP}")
        for schemes_index in range(len(schemes)):
            all_dataset_X = []
            all_dataset_Y = []
            path = "data_vary/AAP=" + str(AAP) + "/trainning_data/" + schemes[schemes_index]
            file_list = [f for f in os.listdir(path) if not f.startswith('.')]
            if len(file_list) == 0:
                print("!! "+schemes[schemes_index]+" No File")
                continue
            for file_name in file_list:
                print("data_vary/AAP=" + str(AAP) + "/trainning_data/" + schemes[schemes_index] + "/"+file_name)
                the_file = open("data_vary/AAP=" + str(AAP) + "/trainning_data/" + schemes[schemes_index] + "/"+file_name, "rb")
                [ML_x_data_all_result, ML_y_data_all_result] = pickle.load(the_file)
                the_file.close()


                for key in ML_x_data_all_result.keys():
                    for dataset_x in ML_x_data_all_result[key]:
                        all_dataset_X.append(dataset_x)
                    for dataset_y in ML_y_data_all_result[key]:
                        all_dataset_Y.append(dataset_y)

            X_train, X_test, y_train, y_test = train_test_split(all_dataset_X, all_dataset_Y,
                                                                test_size=0.1, random_state=1)

            # model = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', algorithm='brute', n_jobs=-1).fit(X_train, y_train)
            model = tree.DecisionTreeClassifier().fit(X_train, y_train)

            y_predict = model.predict(X_test)
            R2_predict = r2_score(y_test, y_predict)
            print(f"R2_predict {R2_predict}")
            MSE_predict = mean_squared_error(y_test, y_predict)
            print(f"MSE_predict {MSE_predict}")

            # save trained model
            os.makedirs("data_vary/AAP=" + str(AAP) + "/trained_ML_model", exist_ok=True)
            the_file = open("data_vary/AAP=" + str(AAP) + "/trained_ML_model/trained_classi_model_"+schemes[schemes_index]+".pkl", "wb+")
            pickle.dump(model, the_file)
            the_file.close()

def train_ML_predict_action_vary_VUB(schemes, x_length, n_neighbors, strategy_number):
    VUB_list = np.array(range(1, 5 + 1)) * 2

    for VUB in VUB_list:
        print(f"VUB={VUB}")
        for schemes_index in range(len(schemes)):
            all_dataset_X = []
            all_dataset_Y = []
            path = "data_vary/VUB=" + str(VUB) + "/trainning_data/" + schemes[schemes_index]
            file_list = [f for f in os.listdir(path) if not f.startswith('.')]
            if len(file_list) == 0:
                print("!! "+schemes[schemes_index]+" No File")
                continue
            for file_name in file_list:
                print("data_vary/VUB=" + str(VUB) + "/trainning_data/" + schemes[schemes_index] + "/"+file_name)
                the_file = open("data_vary/VUB=" + str(VUB) + "/trainning_data/" + schemes[schemes_index] + "/"+file_name, "rb")
                [ML_x_data_all_result, ML_y_data_all_result] = pickle.load(the_file)
                the_file.close()


                for key in ML_x_data_all_result.keys():
                    for dataset_x in ML_x_data_all_result[key]:
                        all_dataset_X.append(dataset_x)
                    for dataset_y in ML_y_data_all_result[key]:
                        all_dataset_Y.append(dataset_y)

            X_train, X_test, y_train, y_test = train_test_split(all_dataset_X, all_dataset_Y,
                                                                test_size=0.1, random_state=1)
            print("y_train", y_train)
            # model = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', algorithm='brute', n_jobs=-1).fit(X_train, y_train)
            model = tree.DecisionTreeClassifier().fit(X_train, y_train)

            y_predict = model.predict(X_test)
            R2_predict = r2_score(y_test, y_predict)
            print(f"R2_predict {R2_predict}")
            MSE_predict = mean_squared_error(y_test, y_predict)
            print(f"MSE_predict {MSE_predict}")

            # save trained model
            os.makedirs("data_vary/VUB=" + str(VUB) + "/trained_ML_model", exist_ok=True)
            the_file = open("data_vary/VUB=" + str(VUB) + "/trained_ML_model/trained_classi_model_"+schemes[schemes_index]+".pkl", "wb+")
            pickle.dump(model, the_file)
            the_file.close()


if __name__ == '__main__':
    # schemes = ["DD-IPI", "DD-ML-IPI", "DD-Random-IPI"]
    schemes = ["DD-IPI", "DD-PI"]
    window_size = 5
    n_neighbors = 50 #50
    strategy_number = 8
    # train_ML_Belief_to_Belief(schemes,window_size,n_neighbors, strategy_number)
    # train_ML_Action_to_Belief(schemes,window_size,n_neighbors, strategy_number)
    # display_prediction_result(schemes, strategy_number)
    x_length = 47

    # ML predict action
    classi_schemes = ["ML_collect_data_PI", "ML_collect_data_IPI"]
    train_ML_predict_action(classi_schemes, x_length, n_neighbors, strategy_number)
    # train_ML_predict_action_vary_AAP(classi_schemes, x_length, n_neighbors, strategy_number)
    # train_ML_predict_action_vary_VUB(classi_schemes, x_length, n_neighbors, strategy_number)
    print('The scikit-learn version is {}.'.format(sklearn.__version__))







