from lib import *


plt.rcParams["figure.figsize"] = (12, 4)
with open('./exp/training_curve_data.json', 'r') as f:
    training_curve_data = json.load(f)

train_loss, val_loss, train_acc, val_acc = [], [], [], []
idx_epochs = list(training_curve_data.keys())

for key, value in training_curve_data.items():
    train_loss.append(value['epoch_train_loss'])
    val_loss.append(value['epoch_val_loss'])
    train_acc.append(value['epoch_accuracy'])
    val_acc.append(value['epoch_val_accuracy'])

plt.plot(idx_epochs, train_loss, label = "Train loss")
plt.plot(idx_epochs, val_loss, label = "Validation loss")
plt.legend()
plt.title('MLP-Mixer Loss Curve')
plt.show()

plt.plot(idx_epochs, train_acc, label = "Train Accuracy")
plt.plot(idx_epochs, val_acc, label = "Validation Accuracy")
plt.legend()
plt.title('MLP-Mixer Accuracy Curve')
plt.show()