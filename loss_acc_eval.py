#%%
import os
import json
import matplotlib.pyplot as plt

# Use the science style (ensure you have installed scienceplots via pip)
plt.style.context(['science', 'nature'])

# Find directories starting with "fine_tuned_model_"
dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith("fine_tuned_model_")]

# Loop over each directory, load the train_metrics JSON file, and extract data
for d in dirs:
    # Extract the model name by removing the prefix
    model_name = d[len("fine_tuned_model_"):]
    json_path = os.path.join(d, "train_metrics.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Assuming each JSON file is a list of dictionaries containing "epoch" and "loss"
        epochs = [entry["epoch"] for entry in data]
        loss = [entry["loss"] for entry in data]
        
        plt.plot(epochs, loss, label=model_name)
    else:
        print(f"Warning: JSON file not found in directory {d}")

# Label the axes and the plot
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Hiba [-]", fontsize=14)
plt.title("Tanítási hiba változása a tanítás során", fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# import os
# import json
# import matplotlib.pyplot as plt

# plt.style.use(['science', 'nature'])

dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith("fine_tuned_model_")]
model_accuracies = {}

for d in dirs:
    model_name = d[len("fine_tuned_model_"):]
    json_path = os.path.join(d, "final_eval_results.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            data = data[0]
        eval_accuracy = data.get("eval_accuracy")
        if eval_accuracy is not None:
            model_accuracies[model_name] = eval_accuracy

# Wrap each accuracy value in a list
accuracy_data = [[acc] for acc in model_accuracies.values()]
labels = list(model_accuracies.keys())

plt.boxplot(
    accuracy_data,
    labels=labels,
    showmeans=True,
    showfliers=True,
    widths=0.3,  # Make the boxes narrower
    meanprops={"marker":"o","markerfacecolor":"green","markeredgecolor":"black","markersize":"5"},
    
)
# plt.xticks(rotation=45)
plt.ylabel("Tanítási pontosság [-]", fontsize=14)
plt.title("Tanítási pontosságok összehasonlítása", fontsize=16)
plt.tight_layout()
plt.show()
# %%
