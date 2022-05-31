import numpy as np
import matplotlib.pyplot as plt


#############################################
########## Run 1 - Architecture #############
#############################################
arr = np.load("data/grid_search_architecture.npy")

learning_rate = [0.00001, 0.0001, 0.001, 0.01]
num_neurons = [24, 48, 72, 96, 120, 144]
architecture = ["LSTM", "LSTM_attention", "GRU", "GRU_attention"]

results = []
x = []
y = []
z = []

for i in range(len(learning_rate)):
    for j in range(len(num_neurons)):
        for k in range(len(architecture)):
            x.append(i)
            y.append(j)
            z.append(k)
            results.append(arr[i,j,k])

ax = plt.axes(projection ="3d")
im = ax.scatter3D(z, x, y, c=results)
cbar = plt.colorbar(im)
cbar.set_label("Validation Loss", rotation=270, labelpad=15)
ax.set_yticks([0, 1, 2, 3], ["0.00001", "0.0001", "0.001", "0.01"])
ax.set_zticks([0, 1, 2, 3, 4, 5], [24, 48, 72, 96, 120, 144])
ax.set_xticks([0, 1, 2, 3], ["LSTM", "LSTM_attention", "GRU", "GRU_attention"])
fig = plt.gcf()
plt.savefig("plots/learning_rate.png")
plt.clf()

# plot learning rate of 0.001 for all the options
height = np.zeros((len(num_neurons), len(architecture)))
for j in range(len(num_neurons)):
    for k in range(len(architecture)):
        height[j,k] = arr[2,j,k]

im = plt.contourf( [0, 1, 2, 3], [0, 1, 2, 3, 4, 5], height)
cbar = plt.colorbar(im)
cbar.set_label("Validation Loss", rotation=270, labelpad=15, fontsize=12)
ax = plt.gca()
ax.set_xlabel("Architecture", fontsize=12)
ax.set_ylabel("# of Neurons", fontsize=12)
# Note: Cartesian x and y are reversed by contour plot
ax.set_yticks([0, 1, 2, 3, 4, 5], [24, 48, 72, 96, 120, 144], fontsize=12)
ax.set_xticks([0, 1, 2, 3], ["LSTM", "LSTM_attention", "GRU", "GRU_attention"], fontsize=12)
plt.savefig("plots/architecture_grid_0.png")
plt.clf()

# plot learning rate of 0.01 for all the options
height = np.zeros((len(num_neurons), len(architecture)))
for j in range(len(num_neurons)):
    for k in range(len(architecture)):
        height[j,k] = arr[3,j,k]

im = plt.contourf( [0, 1, 2, 3], [0, 1, 2, 3, 4, 5], height)
cbar = plt.colorbar(im)
cbar.set_label("Validation Loss", rotation=270, labelpad=15, fontsize=12)
ax = plt.gca()
ax.set_xlabel("Architecture", fontsize=12)
ax.set_ylabel("# of Neurons", fontsize=12)
# Note: Cartesian x and y are reversed by contour plot
ax.set_yticks([0, 1, 2, 3, 4, 5], [24, 48, 72, 96, 120, 144], fontsize=12)
ax.set_xticks([0, 1, 2, 3], ["LSTM", "LSTM_attention", "GRU", "GRU_attention"], fontsize=12)
plt.savefig("plots/architecture_grid_1.png")
plt.clf()

#############################################
########## Run 2 - Regularization ###########
#############################################
arr = np.load("data/grid_search_reg.npy")

reg_penalty = [0.00001, 0.0001, 0.001, 0.01]
reg_method = ["L2", "L1", "Dropout"]
architecture = ["24-neuron LSTM", "48-neuron GRU", "72-neuron LSTM"]
unregularized_results = [0.035333599895238876, 0.03558529540896416, 0.039577990770339966]

# plot regularization grid for each of the three architectures selected by the first grid search
height = np.zeros((len(reg_penalty), len(reg_method) + 1))
for k in range(len(architecture)):
    for i in range(len(reg_penalty)):
        for j in range(len(reg_method)):
            if reg_method[j] == "Dropout":
                # flipping dropout order since I made a mistake when performing grid search
                height[i,j] = arr[-(i - 3), j, k]
            else:
                height[i,j] = arr[i,j,k]
        # Add results without regularization from previous run for comparison
        height[i,j+1] = unregularized_results[k]

    im = plt.contourf([0, 1, 2, 3], [0, 1, 2, 3], height)
    cbar = plt.colorbar(im)
    cbar.set_label("Validation Loss", rotation=270, labelpad=15, fontsize=12)
    ax = plt.gca()
    ax.set_title(architecture[k], fontsize=15)
    ax.set_xlabel("Regularization Penalty/Dropout Rate", fontsize=12)
    ax.set_ylabel("Regularization Method", fontsize=12)
    # Note: Cartesian x and y are reversed by contour plot
    ax.set_xticks([0, 1, 2, 3], ["0.00001 (0.8)", "0.0001 (0.6)", "0.001 (0.4)", "0.01 (0.2)"], fontsize=12)
    ax.set_yticks([0, 1, 2, 3], ["L2", "L1", "Dropout", "None"], fontsize=12)
    plt.savefig("plots/regularization_grid_" + str(k) + ".png")
    plt.clf()

# Best results:
# 24-layer LSTM: arr[0,1,0] = 0.03360379487276077
# corresponds to L1 regularization with 0.00001 penalty

# 48-layer GRU: arr[2,2,1] = 0.03191393241286278
# corresponds to dropout regularization with 0.8 dropout rate

# 72-layer LSTM: arr[2,2,2] = 0.03112339973449707
# corresponds to dropout regularization with 0.8 dropout rate


#############################################
######## Run 3 - Activation and Loss ########
#############################################
arr = np.load("data/grid_search_eval.npy")
# set RMSE of over 1 to NaN so that they don't overly skew plot results
arr[1,2,2] = np.NaN
arr[2,2,2] = np.NaN

loss_func = ["mse", "mae", "huber"]
act_func = ["tanh", "sigmoid", "relu"]
architecture = ["24-neuron LSTM", "48-neuron GRU", "72-neuron LSTM"]

# plot regularization grid for each of the three architectures selected by the first grid search
height = np.zeros((len(loss_func), len(act_func)))
for k in range(len(architecture)):
    for i in range(len(loss_func)):
        for j in range(len(act_func)):
            height[i,j] = arr[i,j,k]

    im = plt.contourf([0, 1, 2], [0, 1, 2], height, vmin=0, vmax=1, extend='max')
    cbar = plt.colorbar(im)
    cbar.set_label("RMSE", rotation=270, labelpad=15, fontsize=12)
    ax = plt.gca()
    ax.set_title(architecture[k], fontsize=15)
    ax.set_xlabel("Activation Function", fontsize=12)
    ax.set_ylabel("Loss Function", fontsize=12)
    # Note: Cartesian x and y are reversed by contour plot
    ax.set_yticks([0, 1, 2], ["mse", "mae", "huber"], fontsize=12)
    ax.set_xticks([0, 1, 2], ["tanh", "sigmoid", "relu"], fontsize=12)
    plt.savefig("plots/loss_" + str(k) + ".png")
    plt.clf()

# Best results:
# 24-layer LSTM w/ L1: arr[0,1,0] = 0.250028520822525 RMSE
# corresponds to tanh with huber loss

# 48-layer GRU w/ Dropout: arr[1,0,1] = 0.4041500985622406
# corresponds to tanh with MAE loss

# 72-layer LSTM w/ Dropout: arr[0,0,2] = 0.425603985786438
# corresponds to tanh with Huber loss
