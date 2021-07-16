from dependency import *

# download the dataset
dataframe = pd.read_csv("http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv", header=None)
raw_data = dataframe.values
# print(dataframe.head())  # 140 points of time series data
#
# print(dataframe.shape)  # (4998, 141) 140 data points and the label

# the last element contains the labels
# print(raw_data.shape) # (4998, 141)
labels = raw_data[:, -1]

# The other data points are the electrocardiogram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)

# Normalize the data to [0,1]
min_val = tf.reduce_min(train_data)  # store min value
max_val = tf.reduce_max(train_data)  # store max value

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

# will train the autoencoder using only the normal rhythms, which are labeled 1
# Separate the normal and abnormal

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

# plot a normal ECG
plt.figure(0)
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A normal ECG")
plt.savefig("normal_ecg.png")

# plt a abnormal ECG
plt.figure(1)
plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An abnormal ECG")
plt.savefig("abnormal_ecg.png")  # overlapped on the normal one\


# Build the model
class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(140, activation="sigmoid")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae') # mean absolute error

# notice that the autoencoder training on only the normal one but evaluated using the full test set


history = autoencoder.fit(normal_train_data, normal_train_data,
                          epochs = 20,
                          batch_size = 512,
                          validation_data = (test_data, test_data),
                          shuffle=True)

# plot the training history
plt.figure(2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training history")
plt.savefig("Training path.png")





