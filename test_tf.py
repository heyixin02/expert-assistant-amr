from tensorflow.keras.optimizers import Adam
import numpy as np

from model.expert_assistant import Expert_Assistant
from utils.dataset import create_datasets

import matplotlib.pyplot as plt
import os



# Configurations
batch_size = 32

input_length = 128
num_classes = 11

data_path = "./dataset/RML2016.10a_dict.pkl"
save_path = "./logs/log2025-08-11-17-16"


# Load datasets
(x_train, y_train), (x_val, y_val), (x_test, y_test), snr = create_datasets(
    num_classes = num_classes, 
    data_path = data_path, 
    save_path = save_path, 
    idx_path = save_path
)

if save_path[-1] != '/':
    save_path += '/'

model_creater = Expert_Assistant
model = model_creater(input_length = input_length, num_classes = num_classes)
model.compile(
    optimizer = Adam(), 
    metrics = ['accuracy'], 
    loss = "categorical_crossentropy"
)
model.build(input_shape = (None, 2, input_length))

# Load the best parameter
model.load_weights(save_path + "best_model.h5")

# Evaluate model
## Overall accuracy
loss, score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score)

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