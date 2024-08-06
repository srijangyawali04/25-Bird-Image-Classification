from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import numpy as np
import torchvision.transforms as tt
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# LOAD THE MODEL (PLEASE USE SCRIPTED VERSION OF MODEL WHICH IS PROVIDED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(r'Model\model_scripted.pt')  # Load your model
model.to(device)

# TEST DATA SET LOCATION
test_dir = r'E:\COMPETETION\Seen_Datasets\val' # THE FOLLOWING TEST IS PERFORMED ON THE PROVIDED VALIDATION SET FROM SEEN DATASET

# SET THE IMAGE SIZE TO 416*416
image_size = (416, 416)

# THIS MEAN AND STANDARD DEVIATION IS CALCULATED FROM TRAINING AND VALIDATION SET SO MIGHT NOT BE ACCURATE FOR TEST SET
mean = [0.4724, 0.4814, 0.4018]
std = [0.2450, 0.2429, 0.2691]

# TRANSFORMING THE TEST SET
transformations_to_perform = tt.Compose([
    tt.Resize(image_size),
    tt.ToTensor(),
    tt.Normalize(mean, std)
])
test_dataset = ImageFolder(test_dir, transformations_to_perform)

# TEST DATA LOADING
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # DO ADJUST BATCH SIZE ACCORDING TO YOUR SYSTEM 

# GENERATES PREDICTION AND GATHERS TRUE LABELS FOR THE TEST SET 
def prediction(test_loader, model, device):
    model.eval()
    list_of_prediction = []
    list_of_probabilities = []
    labels = []
    
    with torch.no_grad():
        for imgs, label in tqdm(test_loader, desc="Processing", leave=False):
            imgs, label = imgs.to(device), label.to(device)

            output = model(imgs)
            if isinstance(output, tuple):
                output = output[0]  

            list_of_probabilities.extend(output.cpu().numpy())
            list_of_prediction.extend(output.argmax(dim=1).cpu().numpy())  
            labels.extend(label.cpu().numpy())  

    return np.array(list_of_prediction), np.array(labels), np.array(list_of_probabilities)

# PREDICTION 
predictions, test_labels, probabilities = prediction(test_loader, model, device)

# REPORT OF PREDICTION
class_names = test_dataset.classes  
report = classification_report(test_labels, predictions, target_names=class_names)
print(report)

# CONFUSION MATRIX 
cm_matrix = confusion_matrix(test_labels, predictions) 
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_matrix, display_labels=class_names)

# CALCULATE ROC-AUC
n_classes = len(class_names)
test_labels_one_hot = label_binarize(test_labels, classes=range(n_classes))

# Calculate ROC-AUC FOR EACH CLASS
roc_auc = roc_auc_score(test_labels_one_hot, probabilities, average='macro', multi_class='ovr')
print(f"Macro-averaged ROC-AUC: {roc_auc:.2f}")

# PLOT CONFUSION MATRIX AND ROC CURVES IN ONE IMAGE
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))  # Increase the figure size

# CONFUSION MATRIX
cm_display.plot(ax=ax1)
ax1.set_title("Confusion Matrix")
ax1.set_xticklabels(class_names, rotation=90)

# ROC CURVE
for i in range(n_classes):
    RocCurveDisplay.from_predictions(test_labels_one_hot[:, i], probabilities[:, i], ax=ax2, name=class_names[i])
ax2.set_title("ROC Curves")

# Plot classification report as text
ax3.axis('off')  
classification_report_str = classification_report(test_labels, predictions, target_names=class_names)
ax3.text(0.5, 0.5, classification_report_str, horizontalalignment='center', verticalalignment='center', fontsize=12, family='monospace')

plt.tight_layout()
plt.savefig('confusion_matrix_roc_and_classification_report.png')
plt.show()