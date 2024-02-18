# %%
import datetime
import sys

sys.path.append('c:/Users/Jonah/Documents/git/IT3030-Deep-Learning/Project1/core')

from functions import REGULARIZATION_FUNCTIONS

# %%
from Doodler import gen_standard_cases
from doodle_config_1 import *
from setup import *

rows=50
cols=50
input_dim=rows*cols
assert LAYER_CONFIG["input"] == input_dim
data, target, labels, img_dim, flat = gen_standard_cases(count=GENERATION["cases"],flat=True,rows=rows,cols=cols, show=False,
                                                         #types=['ball','ring','frame','box','flower']
                                                         )

# %%
def prepare_data():
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# %%
network = Network(output_function=ACTIVATION_FUNCTIONS[LAYER_CONFIG["output_function"]],
                    error_function=ERROR_FUNCTIONS[GLOBAL_CONFIG["loss"]],
                    regularization=REGULARIZATION_FUNCTIONS[GLOBAL_CONFIG["wrt"]],
                    regularization_rate=GLOBAL_CONFIG["wreg"]
                  )
input = LAYER_CONFIG["input"]
for layer in LAYER_CONFIG["layers"]:
    learning_rate = layer["lrate"] if layer["lrate"] else GLOBAL_CONFIG["lrate"]
    print("Added layer with {} neurons, {} inputs, {} activation function and {} learning rate".format(layer["size"],input,layer["activation"],learning_rate))
    network.add_layer(Layer(num_inputs=input,
                            num_neurons=layer["size"],
                            activation=ACTIVATION_FUNCTIONS[layer["activation"]],
                            learning_rate=learning_rate,
                            weight_range=layer["weight_range"]
                            )
                      )
    input = layer["size"]

# %%
test_error, training_errors, validation_errors, training_losses, validation_losses, training_accuracy, validation_accuracy = train_model(
    dataset=prepare_data,
    network=network,
    num_epochs=GLOBAL_CONFIG["epochs"],
    batch_size=GLOBAL_CONFIG["batch_size"],
    error_function=ERROR_FUNCTIONS[GLOBAL_CONFIG["loss"]],
    # error_function=lambda y_true, y_pred: np.mean(y_true != y_pred)
)

print("Test error: ", test_error)

# %%
import matplotlib.pyplot as plt
# Plot the accuracy
# plt.title('LR: {}, Layers: {}, Loss: {}, Weight_range: {}'.format(GLOBAL_CONFIG["lrate"],
#                                                 "["+",".join([str(layer["size"]) for layer in LAYER_CONFIG["layers"]])+"]",
#                                                 GLOBAL_CONFIG["loss"],
#                                                 LAYER_CONFIG["layers"][0]["weight_range"]
#                                                 )
#           )
# plt.xlabel('Epoch')
# plt.ylabel('Error')

fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(8, 8))

# Plot training errors
ax1.plot(training_errors)
ax1.plot(validation_errors)
ax1.scatter(len(training_errors), test_error, color='red', label='Test')
ax1.annotate(str(round(test_error, 5)), (len(training_errors), test_error))
ax1.legend(['Training', 'Validation', 'Test'])
ax1.set_title('LR: {}, Lyrs: {}, L: {}, WR: {}, Reg: {}, R_rate: {}'.format(GLOBAL_CONFIG["lrate"],
                                                "["+",".join([str(layer["size"]) for layer in LAYER_CONFIG["layers"]])+"]",
                                                GLOBAL_CONFIG["loss"],
                                                LAYER_CONFIG["layers"][0]["weight_range"],
                                                GLOBAL_CONFIG["wrt"],
                                                GLOBAL_CONFIG["wreg"]
                                                )
          )
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Error')

# Plot training losses
ax2.plot(training_losses)
ax2.plot(validation_losses)
ax2.legend(['Training', 'Validation'])
ax2.set_title('Training Losses')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')

# Plot training accuracy
ax3.plot(training_accuracy)
ax3.plot(validation_accuracy)
ax3.legend(['Training', 'Validation'])
ax3.set_title('Training Accuracy')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig('./Project1/img/'+datetime.datetime.now().strftime("%d_%H_%M_%S")+"_"+'cases-'+str(GENERATION["cases"])+'_batchsize-'+str(GLOBAL_CONFIG["batch_size"])+'.png')
plt.show()


