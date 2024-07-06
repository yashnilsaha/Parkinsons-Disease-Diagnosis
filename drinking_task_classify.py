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
from scipy import signal
from scipy import stats
from scipy.integrate import simps
from scipy.spatial import ConvexHull
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras import backend as K
import seaborn as sns
import warnings
import tensorflow as tf
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

    # calcutate the Spectral entropy.
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
            plt.ylabel('Power spectral density (V^2 / Hz)')
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
    # print(dir(psd))
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

    row = [record_id, position_name]
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

trajectory_file = 'Drinking_all_export.txt'

with open(trajectory_file, 'r') as infile:
    comm_dict = json.load(infile)

print(len(comm_dict.keys())) # there are 387 keys
print(comm_dict['26'].keys())# prints 'position' and 'resp'
print(sorted(comm_dict['26']['position'].keys())) # prints ['Lank', 'Lelb', 'Lhip', 'Lkne', 'Lsho', 'Lwri', 'Rank', 'Relb', 'Rhip', 'Rkne', 'Rsho', 'Rwri', 'face', 'head', 'neck']

plt.plot(np.array(comm_dict['52']['position']['Lsho'])[:,0]) # visualizing the horizontal movement
plt.show()
part = "Lank"
horizontal_displacemt_array = np.array(comm_dict['26']['position'][part])[:,0]
vertical_displacement_array = np.array(comm_dict['26']['position'][part])[:,1]
fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(121)
ax.plot(horizontal_displacemt_array,label='Horizontal motion')
ax.legend(loc='best')

ax2 = fig.add_subplot(122)
ax2.plot(vertical_displacement_array,label='vertical motion')
ax2.legend(loc='best')

plt.suptitle('Joint motion')
#plt.show()

record_df = pd.DataFrame(columns=["combine_record_id","position_name",
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

#Unified Dyskinesia Rating Scale
rating_file = 'UDysRS.txt'

with open(rating_file, 'r') as infile:
    ratings = json.load(infile)

print(ratings.keys())

print(ratings['Drinking']['2'])

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
        rating = ratings['Drinking'][str(record_id)][order[sub_group]]
    except:
        rating = 1
    return (rating)

def subscore_map(argument):
    order = {"Neck": 0,
             "Rarm": 1,
             "Larm": 2,
             "Trunk": 3,
             "Rleg": 4,
             "Lleg": 5}
    return order.get(argument, 0)


def reverse_lookup_subscore_map(argument):
    order = {0: "Neck",
             1: "Rarm",
             2: "Larm",
             3: "Trunk",
             4: "Rleg",
             5: "Lleg"}
    return order.get(argument, "Neck")

def plot_roc_curve(fpr, tpr, joint):
  plt.plot(fpr,tpr)
  plt.axis([0,1,0,1])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(joint + ' ROC curve')
  plt.show()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

for record_id, group in groups:
    # print(record_id)
    for index, dict_ in group.iterrows():
        position_name = dict_["position_name"]
        for sub_score, values in sub_score_dict.items():
            if position_name in values:
                # print(key, position_name)
                dict_["UDysRS_rating"] =find_rating(dict_["combine_record_id"], sub_score)
                subscore_val = subscore_map(sub_score.strip())
                dict_["sub_score"] = subscore_val
                UDysRS_ratingVal = dict_["UDysRS_rating"]
                #if (UDysRS_ratingVal) < 1.0 :
                #if (UDysRS_ratingVal >0.5 and UDysRS_ratingVal < 1.0 ) or  (UDysRS_ratingVal > 1.5 and  UDysRS_ratingVal<= 2.0):
                if (UDysRS_ratingVal) <=0.5:
                    dict_["UDysRS_rating"] = 0
                else:
                    dict_["UDysRS_rating"] = 1
                processed_df = processed_df.append(dict_, ignore_index=True)


#print("Histogram plot: ")
#processed_df.hist( column=["UDysRS_rating"],figsize=(8,6))
#plt.show()
print(processed_df.head())
print('count of rows in drinking task ', processed_df.shape[0])

#mean of sub score
#grouped_df = processed_df.groupby(['combine_record_id', 'sub_score']).mean().reset_index()
#grouped_df = processed_df.groupby(['combine_record_id', 'sub_score'],as_index=False).mean()
grouped_df = processed_df
print(grouped_df.head(7))


print("****Classification using Neural Network****")
sub_score_gr = grouped_df.groupby(["sub_score"])
for sub_score, sub_score_group in sub_score_gr:
    #print(sub_score)
    print("Classification Details for Joint: " , reverse_lookup_subscore_map(sub_score))
    joint = reverse_lookup_subscore_map(sub_score)
    y = sub_score_group["UDysRS_rating"].astype('float64')
    # print(y.shape)
    X = sub_score_group[
        ["sub_score", "convexhull", "speed_max", "speed_median", "speed_mean", "speed_std_div", "speed_iqr_range",
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
         "velocity_power_bands_0_to_4", "velocity_power_bands_0_to_6"]]
    #print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    print(tf.__version__)

    x_array_train = np.asarray(X_train).astype(np.float32)
    y_array_train = np.asarray(y_train).astype(np.float32)
    x_array_test = np.asarray(X_test).astype(np.float32)
    y_array_test = np.asarray(y_test).astype(np.float32)


    model_4 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_4.compile(loss=tf.keras.losses.binary_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(lr=0.1),
                    metrics=['accuracy', f1_m, tf.keras.metrics.Precision(name='precision'),
                             tf.keras.metrics.Recall(name='recall')])

    history = model_4.fit(x_array_train, y_array_train, epochs=15, batch_size=128, verbose=0, validation_split=0.2)

    # Evaluate the model
    loss, accuracy, f1_score, pre_score, recall_score = model_4.evaluate(x_array_test, y_array_test)
    print(f' Model loss on the test set: {loss}')
    print(f' Model accuracy on the test set: {100 * accuracy}')
    print(f' Model F1 score on the test set: {f1_score}')
    print(f' Model precision score on the test set: {pre_score}')
    print(f' Model recall score on the test set: {recall_score}')


    # model predict
    y_pred_keras = model_4.predict(x_array_test).ravel()

    # extract the predicted probabilities
    p_pred_keras = y_pred_keras.flatten()
    print(p_pred_keras.round(2))

    # extract the predicted class labels
    p_pred_keras_extract = np.where(p_pred_keras > 0.5, 1, 0)
    print('Y predict keras', p_pred_keras_extract)
    print('Y Test ', y_test)
    # ROC illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
    # The critical point here is "binary classifier" and "varying threshold".
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, p_pred_keras_extract)
    auc_keras = auc(fpr_keras, tpr_keras)

    auc_score = roc_auc_score(y_test, p_pred_keras_extract)
    print('AUC score ', auc_score)

    #
    # Calculate the confusion matrix
    #
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=p_pred_keras_extract)
    print(confusion_matrix(y_test, p_pred_keras_extract))
    print(classification_report(y_test, p_pred_keras_extract))

    sns.heatmap(conf_matrix, annot=True, fmt='g')
    plt.xlabel('Predictions')
    plt.ylabel('Actuals')
    plt.title('Drinking: Confusion Matrix for '+joint, fontsize=18)
    #plt.show()

    acc = history.history['f1_m']
    val_acc = history.history['val_f1_m']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training f1-score')
    plt.plot(epochs, val_acc, 'g', label='Validation f1-score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-score')
    plt.title('Model training for the joint ' + joint + ' in drinking task')
    plt.legend()
    fig = plt.figure()
    fig.savefig('acc.png')

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')

    plt.legend()
    #plt.show()

    # plot the ROC Receiver operating characteristic
    #plot_roc_curve(fpr_keras, tpr_keras, joint)

