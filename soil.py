# To Do:
# add missing categorical features
# add grid search to models

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error


def get_data():
    # Import raw data sets into pd.DataFrame
    train = pd.read_csv('training.csv')
    test = pd.read_csv('sorted_test.csv')

    # Drop the CO2 band per instructions
    train.drop(['m2379.76', 'm2377.83', 'm2375.9', 'm2373.97', 'm2372.04', 'm2370.11',
                'm2368.18', 'm2366.26', 'm2364.33', 'm2362.4', 'm2360.47', 'm2358.54',
                'm2356.61', 'm2354.68', 'm2352.76'], axis=1, inplace=True)
    test.drop(['m2379.76', 'm2377.83', 'm2375.9', 'm2373.97', 'm2372.04', 'm2370.11',
               'm2368.18', 'm2366.26', 'm2364.33', 'm2362.4', 'm2360.47', 'm2358.54',
               'm2356.61', 'm2354.68', 'm2352.76'], axis=1, inplace=True)

    # Split out the labels and data for the dependent variables, training set, and test sets
    train_predict_data = train[['Ca', 'P', 'pH', 'SOC', 'Sand']].values
    train_predict_labels = train[['Ca', 'P', 'pH', 'SOC', 'Sand']].columns.values

    train_feature_data = train.ix[:, :3579].values  # includes PIDN
    train_feature_labels = train.ix[:, :3579].columns.values
    print train_feature_labels[3500:]
    print train_feature_labels.shape

    test_feature_data = np.array(test.ix[:, :3579])  # includes PIDN

    # Split data in to test/train sets
    # Since train_test_split doesn't like a matrix of dependent variables we have to do this once for each
    ca_x_train, ca_x_test, ca_y_train, ca_y_test = train_test_split(train_feature_data, train_predict_data[:, 0], test_size=0.33, random_state=42)
    p_x_train, p_x_test, p_y_train, p_y_test = train_test_split(train_feature_data, train_predict_data[:, 1], test_size=0.33, random_state=42)
    ph_x_train, ph_x_test, ph_y_train, ph_y_test = train_test_split(train_feature_data, train_predict_data[:, 2], test_size=0.33, random_state=42)
    soc_x_train, soc_x_test, soc_y_train, soc_y_test = train_test_split(train_feature_data, train_predict_data[:, 3], test_size=0.33, random_state=42)
    sand_x_train, sand_x_test, sand_y_train, sand_y_test = train_test_split(train_feature_data, train_predict_data[:, 4], test_size=0.33, random_state=42)

    # Now that the data have been shuffled and split, we can send the PIDNs along separately
    # Extract them and strip them off the data sets on the way out
    pidn_train = ca_x_train[:, 0]
    pidn_test = ca_x_test[:, 0]

    data = {
        'Ca': {'X_train': ca_x_train[:, 1:], 'X_test': ca_x_test[:, 1:], 'y_train': ca_y_train, 'y_test': ca_y_test},
        'P': {'X_train': p_x_train[:, 1:], 'X_test': p_x_test[:, 1:], 'y_train': p_y_train, 'y_test': p_y_test},
        'pH': {'X_train': ph_x_train[:, 1:], 'X_test': ph_x_test[:, 1:], 'y_train': ph_y_train, 'y_test': ph_y_test},
        'SOC': {'X_train': soc_x_train[:, 1:], 'X_test': soc_x_test[:, 1:], 'y_train': soc_y_train, 'y_test': soc_y_test},
        'Sand': {'X_train': sand_x_train[:, 1:], 'X_test': sand_x_test[:, 1:], 'y_train': sand_y_train, 'y_test': sand_y_test},
    }
    return data, train_feature_labels, train_predict_labels, test_feature_data, pidn_train, pidn_test  # test_feature_data includes PIDN


def get_models():
    # Create models for each dependent variable
    model_list = {
        'Ca': svm.SVR(C=10000.0, verbose=2),
        'P': svm.SVR(C=10000.0, verbose=2),
        'pH': svm.SVR(C=10000.0, verbose=2),
        'SOC': svm.SVR(C=10000.0, verbose=2),
        'Sand': svm.SVR(C=10000.0, verbose=2)
    }
    return model_list

if __name__ == '__main__':
    data_sets, feature_labels, output_labels, test_data, train_ids, test_ids = get_data()
    models = get_models()

    predictions = {}
    output = pd.DataFrame()
    output['PIDN'] = test_data[:, 0]

    for var in output_labels:

        models[var].fit(data_sets[var]['X_train'], data_sets[var]['y_train'])
        predictions = models[var].predict(data_sets[var]['X_test'])

        # Print some evaluation metrics for us
        print str(var) + " min: " + str(np.min(data_sets[var]['y_test']))
        print str(var) + " max: " + str(np.nanmax(data_sets[var]['y_test']))
        print str(var) + " variance: " + str(np.var(data_sets[var]['y_test']))
        print "RMSE: " + str(mean_squared_error(data_sets[var]['y_test'], predictions))
        print ""

        # Write out predictions to file
        #output[var] = models[var].predict(test_data)



    #output.to_csv('beating_benchmark_test.csv', index = False)