import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np

from model.expert_assistant import Expert_Assistant
from utils.dataset import create_datasets

import matplotlib.pyplot as plt
from datetime import datetime
import os



# Configurations
batch_size = 32

input_length = 128
num_classes = 11

data_path = "./dataset/RML2016.10a_dict.pkl"
save_path = f"./logs/log" + datetime.now().strftime("%Y-%m-%d-%H-%M")
os.makedirs(save_path, exist_ok = True)


# Load datasets
(x_train, y_train), (x_val, y_val), (x_test, y_test), snr = create_datasets(
    num_classes = num_classes, 
    data_path = data_path, 
    save_path = save_path, 
    idx_path = None
)

# Build model
model_creater = Expert_Assistant
model = model_creater(input_length = input_length, num_classes = num_classes)
model.compile(
    optimizer = Adam(), 
    metrics = ['accuracy'], 
    loss = 'categorical_crossentropy'
)

# Saving checkpoints
if save_path[-1] != '/':
        save_path += '/'
model_save_path = save_path + "model_{epoch:02d}-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(
    model_save_path, 
    monitor = 'val_accuracy', 
    verbose = 0, 
    save_best_only = True, 
    save_weights_only = True
)

callbacks = [checkpoint]

# Train model
history = model.fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = 100,
    verbose = 2,
    validation_data = (x_val, y_val), 
    shuffle = True, 
    callbacks = callbacks
)

# Plot the training curves
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, np.array(history.history['accuracy']), label='train acc')
plt.plot(history.epoch, np.array(history.history['val_accuracy']), label='val acc')
plt.xlabel('epoch')
plt.legend()
plt.savefig(save_path + "train_curve.png")

# Load the best (latest saved) model
best_model_path = None
best_model_mtime = 0
with os.scandir(save_path) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.endswith('.h5'):
                mtime = entry.stat().st_mtime
                if mtime > best_model_mtime:
                      best_model_path = entry.path
                      best_model_mtime = mtime

print("best model: ", best_model_path)
model.load_weights(best_model_path)
model.save_weights(save_path + "best_model.h5") # Save as the best model                  

# Evaluate model
## Overall accuracy
loss, score = model.evaluate(x_test, y_test, verbose=0)
print('Average accuracy:', score)

## Plot the acc-snr curve
pred_labels = model.predict(x_test, verbose = 1)
pred = np.argmax(pred_labels, axis = 1)
true = np.argmax(y_test, axis = 1)
corrects = np.array(pred == true, dtype = 'int')

# ------- Accuracy of SNRs ---------- #
snrs = [x for x in range(-20, 20, 2)]
corrects_snr = np.zeros(len(snrs))
counts_snr = np.zeros(len(snrs))

for i in range(len(snr)):
    #test_i = test_idx[i]
    counts_snr[(snr[i]+20)//2] += 1
    
    if corrects[i] == 1:
        corrects_snr[(snr[i]+20)//2] += 1

accs_snr = corrects_snr / counts_snr
print(f"Highest accuracy: {np.amax(accs_snr):.4f}")
print("Acc-snr curve: ", accs_snr.tolist())

plt.figure()
plt.plot(snrs, accs_snr, marker = '^')
plt.title('acc-SNR curve')
plt.grid()
plt.savefig(save_path + "acc_snr_curve.png")

# Save test logs
f = open(save_path + "model_test_logs.txt", 'w')
f.write(model_creater.__name__ + '\n')
f.write(f'Average accuracy: {score:.4f}\n')
f.write(f"Highest accuracy: {np.amax(accs_snr):.4f}\n")
f.close()