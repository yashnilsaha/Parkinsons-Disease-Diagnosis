try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy import stats
from scipy.integrate import simps
from scipy.spatial import ConvexHull
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.losses import MeanSquaredLogarithmicError
from keras.models import Sequential
from keras import backend as K
import warnings
warnings.filterwarnings("ignore")
sns.set(font_scale=1.2)


def combined_horizontal_and_vertical(horizontal_array, vertical_array):
    combined_array = np.sqrt(np.square(horizontal_array) + np.square(vertical_array))
    return combined_array


def convert_into_velocity(displacement_array, plot=False):
    velocity_array = np.diff(displacement_array)
    return velocity_array


def convert_into_acceleration(velocity_array, plot=False):
    acceleration_array = np.diff(velocity_array)
    return acceleration_array


def convert_into_jerk(acceleration_array, plot=False):
    jerk_array = np.diff(acceleration_array)
    return jerk_array


def get_kinetic_feature(motion):
    max_motion = np.amax(motion, axis=0)
    median_motion = np.median(motion)
    mean_motion = np.mean(motion, axis=0)
    standard_division_motion = np.std(motion)
    IQR_range = stats.iqr(motion, interpolation = 'midpoint')
    return [max_motion, median_motion, mean_motion, standard_division_motion, IQR_range]


# calculate the Area of Convex Hull of joint movement
def get_convexhull(darray):
    hull = ConvexHull(darray)
    return hull.area


def get_spectral_feature(signals, sample_frequency=10, is_plot=False):
    sf = sample_frequency
    win = 4 * sf

    # calculate the Spectral entropy.
    def spectral_entropy(psd, normalize=False):
        psd_norm = np.divide(psd, psd.sum())
        se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
        if normalize:
            se /= np.log2(psd_norm.size)
        return se

    # calculate the power band for given frequency.
    def bandpower(psd, freqs, min_freqs, max_freqs, is_plot=False):
        # Define delta lower and upper limits
        low, high = min_freqs, max_freqs

        # Find intersecting values in frequency vector
        idx_delta = np.logical_and(freqs >= low, freqs <= high)

        if is_plot:
            # Plot the power spectral density and fill the delta area
            plt.figure(figsize=(7, 4))
            plt.plot(freqs, psd, lw=2, color='k')
            plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power spectral density (uV^2 / Hz)')
            plt.xlim([0, 10])
            plt.ylim([0, psd.max() * 1.1])
            plt.title("Welch's periodogram")
            sns.despine()

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

        # Compute the absolute power by approximating the area under the curve
        delta_power = simps(psd[idx_delta], dx=freq_res)
        # print('Absolute delta power: %.3f uV^2' % delta_power)
        return delta_power

    freqs, psd = signal.welch(signals, sf, nperseg=win)
    if is_plot:
        sns.set(font_scale=1.2, style='white')
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd, color='k', lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.ylim([0, psd.max() * 1.1])
        plt.title("Welch's periodogram")
        plt.xlim([0, freqs.max()])
        sns.despine()

    features = {}
    features["peak_magnitude"] = np.sqrt(psd.max())
    features["entropy"] = spectral_entropy(psd)
    features["half_point"] = freqs.mean()

    features["total_power"] = bandpower(psd, freqs, freqs.min(), freqs.max(), is_plot)
    features["power_bands_0.5_to_1"] = bandpower(psd, freqs, 0.5, 1, is_plot)
    features["power_bands_0_to_2"] = bandpower(psd, freqs, 0, 2, is_plot)
    features["power_bands_0_to_4"] = bandpower(psd, freqs, 0, 4, is_plot)
    features["power_bands_0_to_6"] = bandpower(psd, freqs, 0, 6, is_plot)
    return features



def record_convertion(position_array, position_name, record_id="1-1"):
    position_array = np.array(position_array)
    horizontal_position = position_array[:, 0]
    vertical_position = position_array[:, 1]
    displacement_array = combined_horizontal_and_vertical(horizontal_position, vertical_position)
    velocity_array = convert_into_velocity(displacement_array)
    acceleration_array = convert_into_acceleration(velocity_array)
    jerk_array = convert_into_jerk(acceleration_array)
    record = record_id.split("-")

    row = [record_id, int(record[0]), record[1], position_name]
    row.extend(get_kinetic_feature(velocity_array))
    row.extend(get_kinetic_feature(acceleration_array))
    row.extend(get_kinetic_feature(jerk_array))
    spectral_feature_displacement = get_spectral_feature(displacement_array)
    row.extend([value for key, value in spectral_feature_displacement.items()])
    spectral_feature_velocity = get_spectral_feature(velocity_array)
    row.extend([value for key, value in spectral_feature_velocity.items()])
    convex_hull = get_convexhull(position_array)
    row.extend([convex_hull])
    return row

trajectory_file = 'Communication_all_export.txt'

with open(trajectory_file, 'r') as infile:
    comm_dict = json.load(infile)

print(len(comm_dict.keys()))  # there are 387 keys
print(comm_dict['26-1'].keys())  # prints 'position' and 'resp'
print(sorted(comm_dict['26-1']['position'].keys()))   # prints ['Lank', 'Lelb', 'Lhip', 'Lkne', 'Lsho', 'Lwri', 'Rank', 'Relb', 'Rhip', 'Rkne', 'Rsho', 'Rwri', 'face', 'head', 'neck']

plt.plot(np.array(comm_dict['52-3']['position']['Lsho'])[:,0]) # visualizing the horizontal movement
plt.suptitle('Communication task showing the horizontal movement for record id 52-3')
plt.show()

part = "Lank"
horizontal_displacement_array = np.array(comm_dict['26-1']['position'][part])[:,0]
vertical_displacement_array = np.array(comm_dict['26-1']['position'][part])[:,1]
fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(121)
ax.plot(horizontal_displacement_array,label='Horizontal Left Ankle Motion for record 26-1')
ax.legend(loc='best')

ax2 = fig.add_subplot(122)
ax2.plot(vertical_displacement_array,label='Vertical Left Ankle motion for record 26-1')
ax2.legend(loc='best')

plt.suptitle('Joint motion for dyskinesia in communication task')
plt.show()

record_df = pd.DataFrame(columns=["combine_record_id","record_id", "term","position_name",
                                  "speed_max", "speed_median", "speed_mean", "speed_std_div", "speed_iqr_range",
                                  "acceleration_max", "acceleration_median", "acceleration_mean", "acceleration_std_div", "acceleration_iqr_range",
                                  "jerk_max", "jerk_median", "jerk_mean", "jerk_std_div", "jerk_iqr_range",
                                  "displacement_peak_magnitude","displacement_entropy", "displacement_half_point", "displacement_total_power",
                                  "displacement_power_bands_0.5_to_1","displacement_power_bands_0_to_2", "displacement_power_bands_0_to_4", "displacement_power_bands_0_to_6",
                                  "velocity_peak_magnitude","velocity_entropy", "velocity_half_point", "velocity_total_power",
                                  "velocity_power_bands_0.5_to_1","velocity_power_bands_0_to_2", "velocity_power_bands_0_to_4", "velocity_power_bands_0_to_6",
                                  "convexhull"])
index = 0

for record_id, values in comm_dict.items():
    positions = values["position"]
    resp = values["resp"]
    for position_name, position_array in positions.items():
        row = record_convertion(position_array, position_name, record_id)
        record_df.loc[index] = row
        index += 1

print(record_df.head(10))

#UDysRS Unified Dyskinesia Rating Scale
rating_file = 'UDysRS.txt'

with open(rating_file, 'r') as infile:
    ratings = json.load(infile)

print(ratings.keys())

print(ratings['Communication']['2'])

#create sub score of joints
sub_score_dict = {"Neck": ["face"],
                  "Larm": ["Lsho", "Lelb", "Lwri"],
                  "Rarm": ["Rsho", "Relb", "Rwri"],
                  "Trunk": ["Rsho", "Lsho"],
                  "Rleg": ["Rhip", "Rkne", "Rank"],
                  "Lleg": ["Lhip", "Lkne", "Lank"]}

# sub_score_dict
groups = record_df.groupby("combine_record_id")
processed_df = pd.DataFrame(columns=["combine_record_id", "record_id", "term", "position_name", "sub_score",
                                     "speed_max", "speed_median", "speed_mean", "speed_std_div", "speed_iqr_range",
                                     "acceleration_max", "acceleration_median", "acceleration_mean",
                                     "acceleration_std_div", "acceleration_iqr_range",
                                     "jerk_max", "jerk_median", "jerk_mean", "jerk_std_div", "jerk_iqr_range",
                                     "displacement_peak_magnitude", "displacement_entropy", "displacement_half_point",
                                     "displacement_total_power",
                                     "displacement_power_bands_0.5_to_1", "displacement_power_bands_0_to_2",
                                     "displacement_power_bands_0_to_4", "displacement_power_bands_0_to_6",
                                     "velocity_peak_magnitude", "velocity_entropy", "velocity_half_point",
                                     "velocity_total_power",
                                     "velocity_power_bands_0.5_to_1", "velocity_power_bands_0_to_2",
                                     "velocity_power_bands_0_to_4", "velocity_power_bands_0_to_6",
                                     "convexhull", "UDysRS_rating"])


def find_rating(record_id, sub_group):
    order = {"Neck": 0,
             "Rarm": 1,
             "Larm": 2,
             "Trunk": 3,
             "Rleg": 4,
             "Lleg": 5}
    try:
        rating = ratings['Communication'][str(record_id)][order[sub_group]]
    except:
        rating = 0
    return rating



for record_id, group in groups:
    for index, dict_ in group.iterrows():
        position_name = dict_["position_name"]
        for sub_score, values in sub_score_dict.items():
            if position_name in values:
                # print(key, position_name)
                dict_["sub_score"] = sub_score
                dict_["UDysRS_rating"] = find_rating(dict_["record_id"], sub_score)


                processed_df = processed_df.append(dict_, ignore_index=True)


print(processed_df.head())
print('count of rows in communication task ', processed_df.shape[0])



#mean of sub score
grouped_df = processed_df.groupby(['record_id', 'sub_score']).mean().reset_index()
print(grouped_df.head(7))






#Model
print("**** Analysis using Artificial Neural Network using TensorFlow****")
# create ANN model
sub_score_gr = grouped_df.groupby(["sub_score"])
for sub_score, sub_score_group in sub_score_gr:
    print(sub_score)
    y = sub_score_group["UDysRS_rating"].astype('float64')
    X = sub_score_group.drop(['record_id', 'sub_score', 'UDysRS_rating'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


    def pearson_r(y_true, y_pred):
        # use smoothing for not resulting in NaN values
        # pearson correlation coefficient
        epsilon = 10e-5
        x = y_true
        y = y_pred
        mx = K.mean(x)
        my = K.mean(y)
        xm, ym = x - mx, y - my
        r_num = K.sum(xm * ym)
        x_square_sum = K.sum(xm * xm)
        y_square_sum = K.sum(ym * ym)
        r_den = K.sqrt(x_square_sum * y_square_sum)
        r = r_num / (r_den + epsilon)
        return K.mean(r)


    def scale_datasets(x_train, x_test):
        """
          Standard Scale test and train data
          Z - Score normalization
          """
        standard_scaler = StandardScaler()
        x_train_scaled = pd.DataFrame(
            standard_scaler.fit_transform(x_train),
            columns=x_train.columns
        )
        x_test_scaled = pd.DataFrame(
            standard_scaler.transform(x_test),
            columns=x_test.columns
        )
        return x_train_scaled, x_test_scaled


    x_train_scaled, x_test_scaled = scale_datasets(X_train, X_test)
    hidden_units1 = 160
    hidden_units2 = 480
    hidden_units3 = 256
    learning_rate = 0.01


    # Creating model using the Sequential in tensorflow
    def build_model_using_sequential():
        model = Sequential([
            Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
            Dropout(0.2),
            Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
            Dropout(0.2),
            Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
            Dense(1, kernel_initializer='normal', activation='linear')
        ])
        return model
    # build the model
    model = build_model_using_sequential()
    # loss function
    # Mean Squared Logarithmic Loss as loss function and metric, and Adam loss function optimizer
    msle = MeanSquaredLogarithmicError()
    model.compile(
        loss=msle,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[rmse, pearson_r]
    )
    # train the model

    history = model.fit(
        x_train_scaled, y_train,
        epochs=25,
        batch_size=64,
        validation_split=0.2
    )

    def plot_history(history, key):
        plt.plot(history.history[key])
        plt.plot(history.history['val_' + key])
        plt.xlabel("Epochs")
        plt.ylabel(key)
        plt.title('Model training for communication task')
        plt.legend([key, 'validation_' + key])
        plt.show()


    # Plot the history after model is trained

    plot_history(history, 'rmse')
    prediction_test = model.predict(x_test_scaled)
    print('ANN prediction test', prediction_test)


    results = model.evaluate(x_test_scaled, y_test, verbose=0)

    print('ANN  results', results)
    print('ANN loss :', results[0])
    print('ANN rmse :', results[1])
    print('ANN r :', results[2])



