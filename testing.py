import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
import numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd

#data transformations for testing
data_transforms = {
    'testing': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

#  data directory
data_dir = 'D:\\seem project\\SEEM\\data\\pascal_voc_2012'

# data loaders for testing
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['testing']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['testing']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['testing']}

class_names = image_datasets['testing'].classes

# Loading of saved model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
model.load_state_dict(torch.load('object_classification_model.pth'))
model.eval()

# new model with the correct final layer
new_model = models.resnet18(pretrained=True)
new_model.fc = nn.Linear(new_model.fc.in_features, 20)  # Adjust to match the desired output units


new_model.fc.weight.data = model.fc.weight.data[0:20]  # Copy only the first 2 output units
new_model.fc.bias.data = model.fc.bias.data[0:20]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Testing loop
num_epochs = 1  # You can set the number of epochs to 1 since you are just evaluating

def main():
    results_dict = {'Class': [], 'AP': []}

    for epoch in range(num_epochs):
        for phase in ['testing']:
            running_loss = 0.0
            all_labels = []
            all_probs = []

            # Initialize dictionaries to store labels and probabilities for each class
            class_labels = {class_name: [] for class_name in class_names}
            class_probs = {class_name: [] for class_name in class_names}

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = new_model(inputs)
                    probs = torch.softmax(outputs, dim=1)  # Use softmax to get probabilities

                    # Collect overall labels and probabilities
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

                    # Collect class-wise labels and probabilities
                    for i, class_name in enumerate(class_names):
                        class_labels[class_name].extend(labels.cpu().numpy() == i)
                        class_probs[class_name].extend(probs.cpu().numpy()[:, i])

            # Calculate overall mAP
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            mAP = average_precision_score(all_labels, all_probs, average='macro')
            print(f'{phase} Overall mAP: {mAP:.4f}')

            # Calculate class-wise AP
            for class_name in class_names:
                class_labels_arr = np.array(class_labels[class_name])
                class_probs_arr = np.array(class_probs[class_name])
                class_AP = average_precision_score(class_labels_arr, class_probs_arr, average='macro')

                #  results
                print(f'{phase} {class_name} AP: {class_AP:.4f}')

                #  results in the dictionary
                results_dict['Class'].append(class_name)
                results_dict['AP'].append(class_AP)

    #  DataFrame from the results dictionary
    results_df = pd.DataFrame(results_dict)

    # DataFrame to an Excel file
    results_df.to_excel('results.xlsx', index=False)

    print("Results saved to 'results.xlsx'")

if __name__ == '__main__':
    main()
