---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Real-time tracking

SQLiteTracker provides a powerful and flexible way to track computational (e.g., Machine Learning) experiments using an SQLite database.

This tutorial demonstrates training a small network on the Fashion MNIST dataset and tracking the training and validation metrics in real time. We would also see how you can query the tracked metrics while training is ongoing and visualize the metrics vs. epoch plots.

+++

## Create the experiment tracker

```{code-cell} ipython3
:tags: ["hide-cell"]

from pathlib import Path

db = Path('nn_experiments.db')
if db.exists():
    db.unlink()
```
    
```{code-cell} ipython3
from sklearn_evaluation import SQLiteTracker
 
tracker = SQLiteTracker('nn_experiments.db')
experiment = tracker.new_experiment()
uuid = experiment.uuid
```

## MNIST Dataset

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. 

```{code-cell} ipython3
:tags: ["hide-output"]
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

```{code-cell} ipython3
# create a validation set
from sklearn.model_selection import train_test_split

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2)
```

```{code-cell} ipython3
# Inspect an image in the dataset (Pixel values fall in the range 0-255)
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```

```{code-cell} ipython3
train_images = train_images[:500]
train_labels = train_labels[:500]
val_images = val_images[:500]
val_labels = val_labels[:500]
```

```{code-cell} ipython3
# Scale the images to range (0,1)

train_images = train_images / 255.0
val_images = val_images / 255.0
```

## Train the model

```{code-cell} ipython3
# Create all metrics arrays

loss = []
val_loss = []
accuracy = []
val_accuracy = []
```

Define a callback that will track the metrics during the training, and log in the experiment tracker.

```{code-cell} ipython3
class TrackLossandAccuracyCallback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        loss.append(logs["loss"])
        val_loss.append(logs["val_loss"])
        accuracy.append(logs["accuracy"])
        val_accuracy.append(logs["val_accuracy"])
        tracker.upsert(uuid, {"loss": loss, 
                              "accuracy": accuracy, 
                              "val_loss": val_loss, 
                              "val_accuracy": val_accuracy}
                      )
```

```{code-cell} ipython3
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```

```{code-cell} ipython3
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
             )
```

```{code-cell} ipython3
epoch_count = 10
history = model.fit(train_images, 
                    train_labels, 
                    validation_data=(val_images, val_labels), 
                    epochs=epoch_count, 
                    verbose=0, 
                    callbacks=[TrackLossandAccuracyCallback()])
```

## Track metrics while training

While the training is ongoing, you may visualize the metrics plot by opening another terminal/notebook and running the steps below.

+++

Query the experiment with SQL:

```{code-cell} ipython3
results = tracker.query("""
SELECT
    uuid,
    json_extract(parameters, '$.accuracy') as accuracy,
    json_extract(parameters, '$.loss') as loss,
    json_extract(parameters, '$.val_accuracy') as val_accuracy,
    json_extract(parameters, '$.val_loss') as val_loss
    FROM experiments
""")
```

Extract and plot the relevant metrics against epochs:

```{code-cell} ipython3
import json

training_accuracy = json.loads(results["accuracy"].to_list()[0])
val_accuracy = json.loads(results["val_accuracy"].to_list()[0])
training_loss = json.loads(results["loss"].to_list()[0])
val_loss = json.loads(results["val_loss"].to_list()[0])

epoch_range = range(1, len(training_accuracy)+1)
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))                            
ax1.plot(epoch_range, training_loss, color="#725BD0", linestyle="--", label='Train')
ax1.plot(epoch_range, val_loss, color="#725BD0", linestyle="-", label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='best')
ax1.grid()

ax2.plot(epoch_range, training_accuracy, color="#BA2932", linestyle="--", label="Train")
ax2.plot(epoch_range, val_accuracy, color="#BA2932", linestyle="-", label="Validation")
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='best')
ax2.grid()
```
