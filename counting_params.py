#%%
from transformers import AutoModelForSequenceClassification

# Load your model
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-small-en-v1.5")

# Method 1: Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

# Method 2: Count all parameters for comparison
all_params = sum(p.numel() for p in model.parameters())
print(f"All parameters: {all_params:,}")
# %%
def freeze_layers(model, layer_freeze, layer_not_freeze):
    for name, param in model.named_parameters():
        # for freeze in layer_freeze:
        if layer_freeze in name: #!! For testing only embedding layer and attention
            param.requires_grad = False
        # for freeze in layer_not_freeze:

        if layer_not_freeze in name:
            param.requires_grad = True
            # print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")
    for name, param in model.named_parameters():
        print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")

    return model

def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": trainable + frozen,
        "trainable_percent": 100 * trainable / (trainable + frozen)
    }

#%%
# A11
count_parameters(freeze_layers(model, "embedding", "bert.encoder.layer.11"))
# %%
# AC
count_parameters(freeze_layers(model, "bert", "classifier"))

# %%
count_parameters(freeze_layers(model, "bert", "embedding"))

# %%
count_parameters(freeze_layers(model, "bert.embedding", "bert.encoder"))
# %%
