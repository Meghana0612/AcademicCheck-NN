import os
import json

config_path = "models/cross_encoder/train_config.json"

if not os.path.exists(config_path):
    print("No saved training config found.")
    raise SystemExit

with open(config_path, "r") as f:
    config = json.load(f)

print("========== VALIDATION RESULTS ==========\n")

print(f"Validation F1       : {config.get('val_f1')}")
print(f"Validation AUC      : {config.get('val_auc')}")
print(f"Best Threshold      : {config.get('best_threshold')}")
print(f"Base Model          : {config.get('base_model')}")
print(f"Epochs              : {config.get('num_epochs')}")

print("\nValidation Classification Report:\n")
print(config.get("validation_report"))

print("\n========== TEST RESULTS ==========\n")

print(f"Test Accuracy       : {config.get('test_accuracy')}")
print(f"Test F1             : {config.get('test_f1')}")
print(f"Test AUC            : {config.get('test_auc')}")

print("\nTest Classification Report:\n")
print(config.get("test_report"))

print("\n==================================")

print(
    f"\n✔ Done.  Model → models/cross_encoder  |  Threshold → {config.get('best_threshold')}"
)


