#importing important libraries
import os
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import GridSearchCV


# Preprocessing data
img2vec = Img2Vec()

train_dir = r"C:\Users\shhre\OneDrive\Desktop\SML\BloodGroupDataset\train"
val_dir = r"C:\Users\shhre\OneDrive\Desktop\SML\BloodGroupDataset\val"

data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_)

            # Resize and convert to tensor without normalization for grayscale images
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),  # Convert to RGB
                transforms.ToTensor(),
            ])

            img_tensor = preprocess(img)

            # Get features using img2vec
            img_pil = transforms.ToPILImage()(img_tensor)
            img_features = img2vec.get_vec(img_pil)

            features.append(img_features)
            labels.append(category)

    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels



# Convert lists to numpy arrays for sklearn compatibility
data['training_data'] = np.array(data['training_data'])
data['training_labels'] = np.array(data['training_labels'])
data['validation_data'] = np.array(data['validation_data'])
data['validation_labels'] = np.array(data['validation_labels'])



# Print shapes of training data and labels
print('Training data shape:', data['training_data'].shape)
print('Training labels shape:', data['training_labels'].shape)



# Check if training data is empty
if data['training_data'].shape[0] == 0 or data['training_labels'].shape[0] == 0:
    print('Error: Training data or labels are empty. Please check your data.')

else:
    #hyperparamter tuning using GridSearchCV
    svm_param_grid = {'C': [1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    # Train SVC model
    svm_model = SVC(random_state=2)
    svm_grid = GridSearchCV(svm_model, svm_param_grid, cv=5, verbose=1)
    svm_grid.fit(data['training_data'], data['training_labels'])

    print("SVM Best Parameters:", svm_grid.best_params_)
    print("SVM Best Score:", svm_grid.best_score_)

    # Test performance of SVC model
    y_pred_svm = svm_grid.predict(data['validation_data'])
    score_svc = accuracy_score(y_pred_svm, data['validation_labels'])
    print('SVC model accuracy is:', score_svc)

    #For calculate individual accuracy for each blood group
    blood_group_labels = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']
    blood_group_accuracy = {}
    for blood_group in blood_group_labels:
        # Filter validation data and labels for the current blood group
        blood_group_indices = data['validation_labels'] == blood_group
        validation_data_blood_group = data['validation_data'][blood_group_indices]
        validation_labels_blood_group = data['validation_labels'][blood_group_indices]

        # Predict using SVC model
        y_pred_svc_blood_group = svm_grid.predict(validation_data_blood_group)
        accuracy_svc_blood_group = accuracy_score(y_pred_svc_blood_group, validation_labels_blood_group)
        blood_group_accuracy[f'SVC_{blood_group}'] = accuracy_svc_blood_group

    # Print individual accuracies
    print('\nIndividual Accuracies:')
    for blood_group, accuracy in blood_group_accuracy.items():
        print(f'{blood_group}: {accuracy}')

    # Calculate and print classification report for overall model
    print('\nClassification Report - SVM:')
    print(classification_report(data['validation_labels'], y_pred_svm, target_names=blood_group_labels))

    # Plot confusion matrix
    cf = confusion_matrix(y_pred_svm, data['validation_labels'])
    cmd = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=blood_group_labels)
    cmd.plot()
    plt.show()

    # Saving model to desired location
    save_path = r"C:\Users\shhre\OneDrive\Desktop\SML\svc_model.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(svm_model, f)

    print("SVC model saved successfully")
