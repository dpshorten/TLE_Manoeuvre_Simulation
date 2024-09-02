import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory + "/../")
from utilities import (load_tle_data_from_file,
                       propagate_SatelliteTLEData_object,
                       get_np_mean_elements_at_epoch_index,
                       propagate_np_mean_elements,
                       convert_np_keplerian_coordinates_to_cartesian,
                       convert_np_cartesian_coordinates_to_keplerian,
                       set_df_elements_at_epoch_index,
                       SatelliteTLEData,
                       DICT_ELEMENT_NAMES,
                       )

def add_velocity_to_cartesian_elements(np_cartesian_elements,
                                       manoeuvre_type = "r",
                                       delta_v = 0):

    position_cartesian = np_cartesian_elements[:3].astype(np.longdouble)
    velocity_cartesian = np_cartesian_elements[3:].astype(np.longdouble)

    u = position_cartesian / np.linalg.norm(position_cartesian)
    w = np.cross(u, velocity_cartesian)
    w /= np.linalg.norm(w)
    v = np.cross(w, u)

    #direction = random.choice([1.0, -1.0])
    direction = 1.0
    if manoeuvre_type == "i":
        new_velocity_cartesian = velocity_cartesian + direction * delta_v * v
    elif manoeuvre_type == "r":
        new_velocity_cartesian = velocity_cartesian + direction * delta_v * u
    else:
        new_velocity_cartesian = velocity_cartesian + direction * delta_v * w

    # print("intial", velocity_cartesian)
    # print("manoeuvred", new_velocity_cartesian)
    # print(v, u, w)
    # print()
    # quit()

    np_new_cartesian_elements = np.copy(np_cartesian_elements)
    np_new_cartesian_elements[3:] = new_velocity_cartesian

    return np_new_cartesian_elements

def add_manoeuvre_to_mean_elements(dict_shared_parameters,
                                   dict_simulation_parameters,
                                   np_mean_elements,
                                   manoeuvre_type,
                                   satellite_name):
    np_cartesian_elements = convert_np_keplerian_coordinates_to_cartesian(np_mean_elements)
    manoeuvre_index = dict_shared_parameters["simulation manoeuvre types"].index(manoeuvre_type)
    manoeuvre_magnitude = dict_simulation_parameters["simulation manoeuvre magnitudes"][satellite_name][manoeuvre_index]
    np_cartesian_elements = add_velocity_to_cartesian_elements(np_cartesian_elements,
                                                               manoeuvre_type,
                                                               manoeuvre_magnitude
                                                               )

    return convert_np_cartesian_coordinates_to_keplerian(np_cartesian_elements)

def simulate_for_one_satellite_video(dict_shared_parameters,
                               dict_simulation_parameters,
                               satelliteTLEData_satellites,
                               np_residuals,
                               np_observation_cov,
                               manoeuvre_type):
    new_list_of_tle_line_tuples = []
    np_new_mean_elements = np.zeros((dict_simulation_parameters["video epochs to simulate"] *
                                     dict_simulation_parameters["video tle multiplier"], 6))
    np_new_mean_elements_no_noise = np.zeros((dict_simulation_parameters["video epochs to simulate"] *
                                              dict_simulation_parameters["video tle multiplier"], 6))
    np_epochs = np.zeros(dict_simulation_parameters["video epochs to simulate"] *
                         dict_simulation_parameters["video tle multiplier"], dtype="datetime64[us]")
    np_mean_elements = get_np_mean_elements_at_epoch_index(satelliteTLEData_satellites, 0)
    for i in range(dict_simulation_parameters["video epochs to simulate"]):
        tle_line_pair_subsequent = satelliteTLEData_satellites.list_of_tle_line_tuples[i + 10]
        for j in range(dict_simulation_parameters["video tle multiplier"]):

            tle_line_1 = satelliteTLEData_satellites.list_of_tle_line_tuples[i][0]
            subsequent_tle_line_1 = satelliteTLEData_satellites.list_of_tle_line_tuples[i + 10][0]
            new_day = (float(tle_line_1[20:32]) + (j / dict_simulation_parameters["video tle multiplier"]) *
                       (float(subsequent_tle_line_1[20:32]) - float(tle_line_1[20:32])))
            # new_day_string = "{day:03.8}".format(day = new_day)
            new_day_whole_string = "{day:03d}".format(day=int(np.floor(new_day)))
            new_day_fraction_string = "{day:.8f}".format(day=new_day - int(np.floor(new_day)))
            new_day_string = new_day_whole_string + new_day_fraction_string[1:]
            new_tle_line_1 = tle_line_1[:20] + new_day_string + tle_line_1[32:]
            new_tle_tuple = (new_tle_line_1, satelliteTLEData_satellites.list_of_tle_line_tuples[i][1])

            np_mean_elements = propagate_np_mean_elements(np_mean_elements, new_tle_tuple,
                                                          tle_line_pair_subsequent,
                                                          element_set=dict_simulation_parameters["element set"],
                                                          proportion=1 / (dict_simulation_parameters[
                                                                              "video tle multiplier"] - j))
            if j % 20 == 0:
                np_mean_elements[2] += 0.2
            new_list_of_tle_line_tuples.append(new_tle_tuple)
            np_new_mean_elements[i * dict_simulation_parameters["video tle multiplier"] + j, :] = (
                    np_mean_elements + np.random.multivariate_normal(np.zeros(np_mean_elements.shape[0]),
                                                                     np.diag(1e-3 * np.diag(np_observation_cov))))
            np_new_mean_elements_no_noise[i * dict_simulation_parameters["video tle multiplier"] + j,
            :] = np_mean_elements

            epoch_time = satelliteTLEData_satellites.pd_df_tle_data.index[i]
            epoch_time = epoch_time
            epoch_time_subsequent = satelliteTLEData_satellites.pd_df_tle_data.index[i + 1]
            np_epochs[i * dict_simulation_parameters["video tle multiplier"] + j] = (epoch_time +
                                                                                     (j / dict_simulation_parameters[
                                                                                         "video tle multiplier"]) *
                                                                                     (
                                                                                                 epoch_time_subsequent - epoch_time))

    pd_df_new_satellites = pd.DataFrame(data=np_new_mean_elements,
                                        columns=DICT_ELEMENT_NAMES["kepler_6"],
                                        index=np_epochs)
    pd_df_new_satellites_no_noise = pd.DataFrame(data=np_new_mean_elements_no_noise,
                                                 columns=DICT_ELEMENT_NAMES["kepler_6"],
                                                 index=np_epochs)
    pd_df_new_satellites_no_noise.to_pickle("foo.pkl")

    satelliteTLEData_new_satellites = SatelliteTLEData(new_list_of_tle_line_tuples,
                                                       pd_df_new_satellites,
                                                       "kepler_6")
    return satelliteTLEData_new_satellites, []

def simulate_for_one_satellite(dict_shared_parameters,
                               dict_simulation_parameters,
                               satelliteTLEData_satellites,
                               np_residuals,
                               np_observation_cov,
                               manoeuvre_type,
                               satellite_name):

    list_manoeuvre_indices = []
    next_manoeuvre = 10

    while next_manoeuvre < dict_simulation_parameters["epochs to simulate"]:
        list_manoeuvre_indices.append(next_manoeuvre)
        next_manoeuvre += np.random.randint(dict_simulation_parameters["time between manoeuvres"][0],
                                            dict_simulation_parameters["time between manoeuvres"][1])

    np_mean_elements = get_np_mean_elements_at_epoch_index(satelliteTLEData_satellites, 0)
    for i in range(dict_simulation_parameters["epochs to simulate"]):
        tle_line_pair_initial = satelliteTLEData_satellites.list_of_tle_line_tuples[i]
        tle_line_pair_subsequent = satelliteTLEData_satellites.list_of_tle_line_tuples[i + 1]
        np_mean_elements = propagate_np_mean_elements(np_mean_elements, tle_line_pair_initial, tle_line_pair_subsequent,
                                                      element_set = dict_simulation_parameters["element set"])
        index_of_residuals_to_add = np.random.randint(0, np_residuals.shape[0])

        if i in list_manoeuvre_indices:
            np_mean_elements = add_manoeuvre_to_mean_elements(dict_shared_parameters,
                                                              dict_simulation_parameters,
                                                              np_mean_elements,
                                                              manoeuvre_type,
                                                              satellite_name)

        np_mean_elements += (dict_simulation_parameters["empirical errors modifier"] *
                              np_residuals[index_of_residuals_to_add, :])
        # np_mean_elements += np.random.multivariate_normal(np.zeros(np_mean_elements.shape[0]),
        #                                                   dict_simulation_parameters["variance modifier model"] *
        #                                                   np_observation_cov)
        set_df_elements_at_epoch_index(satelliteTLEData_satellites,
                                       i + 1,
                                       np_mean_elements + np.random.multivariate_normal(
                                           np.zeros(np_mean_elements.shape[0]),
                                           dict_simulation_parameters["variance modifier model"] *
                                           np_observation_cov))

    satelliteTLEData_satellites.pd_df_tle_data = (satelliteTLEData_satellites.pd_df_tle_data.
                                                  iloc[:dict_simulation_parameters["epochs to simulate"]])
    satelliteTLEData_satellites.list_of_tle_line_tuples = (satelliteTLEData_satellites.
                                                           list_of_tle_line_tuples[
                                                           :dict_simulation_parameters["epochs to simulate"]])

    pd_series_manoeuvre_timestamps = pd.Series(satelliteTLEData_satellites.pd_df_tle_data.iloc[list_manoeuvre_indices].index)
    list_manoeuvre_datetimes = []
    for i in range(pd_series_manoeuvre_timestamps.shape[0]):
        list_manoeuvre_datetimes.append(pd_series_manoeuvre_timestamps.iloc[i].to_pydatetime())

    return satelliteTLEData_satellites, list_manoeuvre_datetimes

def setup_then_run_simulation(dict_shared_parameters,
                              dict_simulation_parameters,
                              satellite_name,
                              manoeuvre_type,
                              simulation_number,
                              save_directory):

    start_epoch = np.random.randint(dict_simulation_parameters["start epoch"][0], dict_simulation_parameters["start epoch"][1])
    satelliteTLEData_satellites = load_tle_data_from_file(dict_simulation_parameters["original tles directory"] +
                                                          satellite_name +
                                                          dict_simulation_parameters["tle file suffix"],
                                                          start_epoch,
                                                          dict_simulation_parameters["element set"])

    pd_df_propagated_mean_elements = propagate_SatelliteTLEData_object(satelliteTLEData_satellites, 1)
    np_residuals = (pd_df_propagated_mean_elements - satelliteTLEData_satellites.pd_df_tle_data).dropna().values
    residuals_covariance = np.cov(np_residuals, rowvar = False)
    normalisation_weights = np.sqrt(np.diag(residuals_covariance))
    residual_magnitudes = np.sum(np.square(np.divide(np_residuals, normalisation_weights)), axis = 1)
    residual_magnitude_threshold = np.percentile(residual_magnitudes,
                                                 dict_simulation_parameters["empirical errors cutoff percentile"])
    residuals_rows_to_remove = np.argwhere(residual_magnitudes > residual_magnitude_threshold)
    np_residuals = np.delete(np_residuals, residuals_rows_to_remove, axis = 0)

    np_diffs = satelliteTLEData_satellites.pd_df_tle_data.values[1:] - satelliteTLEData_satellites.pd_df_tle_data.values[:-1]
    q1 = np.percentile(np_diffs, [0.25])[0]
    q3 = np.percentile(np_diffs, [0.75])[0]
    iqr = q3 - q1
    np_diffs = np_diffs[~((np_diffs < (q1 - 5 * iqr)) | (np_diffs > (q3 + 5 * iqr))).any(axis=1)]
    np_tle_covariance = (np.cov(np_diffs, rowvar = False))
    if dict_shared_parameters["satellite set"] == "simulated_video":
        satelliteTLEData_satellites, list_manoeuvre_datetimes = simulate_for_one_satellite_video(dict_shared_parameters,
                                                                                           dict_simulation_parameters,
                                                                                           satelliteTLEData_satellites,
                                                                                           np_residuals,
                                                                                           np_tle_covariance,
                                                                                           manoeuvre_type)
    else:
        satelliteTLEData_satellites, list_manoeuvre_datetimes = simulate_for_one_satellite(dict_shared_parameters,
                                                                                           dict_simulation_parameters,
                                                                                           satelliteTLEData_satellites,
                                                                                           np_residuals,
                                                                                           np_tle_covariance,
                                                                                           manoeuvre_type,
                                                                                           satellite_name)

    # fig, axs = plt.subplots(6)
    # for i in range(6):
    #     axs[i].plot(satelliteTLEData_satellites.pd_df_tle_data[DICT_ELEMENT_NAMES[dict_simulation_parameters["element set"]][i]])
    # plt.show()
    # quit()



    pickle.dump(satelliteTLEData_satellites, open(save_directory +
                                                  satellite_name +
                                                  "_" + manoeuvre_type + "_" + str(simulation_number) +
                                                  dict_shared_parameters["simulated data files suffix"],
                                                  'wb'))
    file_for_manoeuvres = open(save_directory +
                               satellite_name +
                               "_" + manoeuvre_type + "_" + str(simulation_number) +
                               dict_shared_parameters["simulated manoeuvre files suffix"],
                               'w')

    yaml.dump({"name": satellite_name,
               "manoeuvre_timestamps": list_manoeuvre_datetimes},
              file_for_manoeuvres)



dict_shared_parameters = yaml.safe_load(open(sys.argv[1], 'r'))
dict_simulation_parameters = yaml.safe_load(open(sys.argv[2], 'r'))

if dict_shared_parameters["satellite set"] == "simulated":
    for satellite_index in range(len(dict_shared_parameters["simulated satellite names list"])):
        for manoeuvre_type in dict_shared_parameters["simulation manoeuvre types"]:
            for simulation_number in range(dict_shared_parameters["simulations per satellites and manoeuvre type"]):
                setup_then_run_simulation(dict_shared_parameters,
                                          dict_simulation_parameters,
                                          dict_shared_parameters["simulated satellite names list"][satellite_index],
                                          manoeuvre_type,
                                          simulation_number,
                                          dict_shared_parameters["simulated data files directory"])

elif dict_shared_parameters["satellite set"] == "simulated_video":
    setup_then_run_simulation(dict_shared_parameters,
                              dict_simulation_parameters,
                              dict_shared_parameters["simulated video satellite names list"][0],
                              'c',
                              1,
                              dict_shared_parameters["simulated video data files directory"])

else:
    print("settings need to specify a simulated satellite set")
