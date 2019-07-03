# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import csv as csv

print(tf.__version__)

train_array = np.array((-200, -200, -200, -200, -200, 1, 1, 1, 0))
test_array = np.array((-200, -200, -200, -200, -200, 1, 1, 1, 0))
train_labels = np.array(0)
test_labels = np.array(0)

with open('data/measurements_1.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        line = [int(row['A']), int(row['B']), int(row['C']), int(row['D']), int(row['E']),
                float(row['X']), float(row['Y']), float(row['Z']),
                int(row['prev_location'])]
        if row['usefor'] == 'test':
            test_array = np.vstack([test_array, line])
            test_labels = np.vstack([test_labels, int(row['location'])])
        else:
            train_array = np.vstack([train_array, line])
            train_labels = np.vstack([train_labels, int(row['location'])])

class_names = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'D4']

train_array = np.delete(train_array, 0, 0)
train_labels = np.delete(train_labels, [0])
test_array = np.delete(test_array, 0, 0)
test_labels = np.delete(test_labels, [0])

# print(train_array)
# print(train_labels)
# print(test_array)
# print(test_labels)
#
# sys.exit()
#
# train_array = train_array / -1
# test_array = test_array / -1
#
# print(len(train_array))
# print(len(train_labels))

model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_array, train_labels, epochs=100)

test_loss, test_acc = model.evaluate(test_array, test_labels)

print('Test accuracy:', test_acc)


predictions = model.predict(test_array)

correct = 0
false = 0
total = 0

results = {
    'A1': (0, 0, 0),
    'A2': (0, 0, 0),
    'A3': (0, 0, 0),
    'A4': (0, 0, 0),
    'B1': (0, 0, 0),
    'B2': (0, 0, 0),
    'B3': (0, 0, 0),
    'B4': (0, 0, 0),
    'C1': (0, 0, 0),
    'C2': (0, 0, 0),
    'C3': (0, 0, 0),
    'C4': (0, 0, 0),
    'D1': (0, 0, 0),
    'D2': (0, 0, 0),
    'D3': (0, 0, 0),
    'D4': (0, 0, 0)
}

for i in range(len(test_array)):
    result = 'false'
    total += 1
    row = results[class_names[test_labels[i]]]
    rowTotal = row[0]
    rowCorrect = row[1]
    rowFalse = row[2]
    rowTotal += 1
    if test_labels[i] == np.argmax(predictions[i]):
        result = 'correct'
        correct += 1
        rowCorrect += 1
    else:
        false += 1
        rowFalse += 1

    results[class_names[test_labels[i]]] = (rowTotal, rowCorrect, rowFalse)
    # if (test_array[i][8] != test_labels[i]):
    #     print('Test Array:', test_array[i], 'Lable:', test_labels[i], ' ', class_names[test_labels[i]],
    #           'Prediction:', np.argmax(predictions[i]), ' ', class_names[np.argmax(predictions[i])], '', result)
    if result == 'false':
        print('Test Array:', test_array[i], 'Lable:', test_labels[i], ' ', class_names[test_labels[i]],
          'Prediction:', np.argmax(predictions[i]), ' ', class_names[np.argmax(predictions[i])], '', result)

print(results)

print('total:', total, ', correct: ', correct, ', false: ', false, 'accuracy: ', correct/total)
