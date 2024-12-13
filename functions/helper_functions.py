import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def min_max_scaling(X: pd.DataFrame):
    return (X - X.to_numpy().min()) / (X.to_numpy().max() - X.to_numpy().min())


def forward(X, weights, biases, activation_func):
    activations = [X]
    input_data = X

    for w, b in zip(weights, biases):
        z = np.dot(input_data, w) + b
        a = activation_func(z)
        activations.append(a)
        input_data = a

    return activations


def backward(activations, weights, y_true, activation_der, loss_der):
    deltas = []

    y_pred = activations[-1]
    delta = loss_der(y_true, y_pred) * activation_der(y_pred)
    deltas.append(delta)

    for i in range(len(weights)-1, 0, -1):
        delta = np.dot(deltas[-1], weights[i].T) \
                * activation_der(activations[i])
        deltas.append(delta)

    deltas.reverse()
    return deltas


def update_weights(weights, biases, activations, deltas, learning_rate):
    for i in range(len(weights)):
        activations_np = np.array(activations[i])
        deltas_np = np.array(deltas[i])
        weights[i] -= learning_rate * np.dot(activations_np.T, deltas_np)

        biases[i] -= learning_rate * np.sum(deltas_np, axis=0, keepdims=True)


def train_test_valid_split(X: pd.DataFrame,
                           y: pd.DataFrame,
                           percents: list = [0.6, 0.3, 0.1],
                           drop: list = None,
                           shuffle: bool = True,
                           random_seed: int = 42):
    X_normilized = min_max_scaling(X)
    data = pd.concat([X_normilized, y], axis=1)
    y_len = y.shape[1]

    if shuffle:
        data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    data_shuffled = data.copy()

    if drop:
        data.drop(drop, axis=1, inplace=True)

    train_size = int(len(data) * percents[0])
    valid_size = int(len(data) * percents[1])

    train_data = data.iloc[:train_size]
    valid_data = data.iloc[train_size:train_size + valid_size]
    test_data = data.iloc[train_size + valid_size:]

    X_train, y_train = train_data.iloc[:, :-y_len], train_data.iloc[:, -y_len:]
    X_valid, y_valid = valid_data.iloc[:, :-y_len], valid_data.iloc[:, -y_len:]
    X_test, y_test = test_data.iloc[:, :-y_len], test_data.iloc[:, -y_len:]

    return data_shuffled, X_train, y_train, X_valid, y_valid, X_test, y_test


def performance_statistics(y_true, y_pred, classes):
    n_classes = len(classes)

    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label, pred_label] += 1

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)

    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    stats_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClass-wise Performance Metrics:")
    print(stats_df)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def generate_letters(img_side, fonts_percentage, alphabet):
    from os import walk
    from PIL import Image, ImageFont, ImageDraw

    fonts_path = '../../fonts'
    fonts = next(walk(fonts_path), (None, None, []))[2]
    fonts_after_trim = fonts[:int(len(fonts) * fonts_percentage)]

    image_size = (img_side, img_side)
    features = [f'x{i}' for i in range(1, img_side**2 + 1)]
    one_hot_encode_labels = [f'is_{letter}' for letter in alphabet]
    df = pd.DataFrame(columns=features + one_hot_encode_labels + ['letter'])
    for font_file in fonts_after_trim:
        font = ImageFont.truetype(f"{fonts_path}/{font_file}", img_side // 1.5)

        for index, letter in enumerate(alphabet):
            image = Image.new('1', image_size, 0)
            draw = ImageDraw.Draw(image)

            draw.text((img_side // 5, img_side // 9.2), letter, font=font, fill=255)

            df = pd.concat(
                [
                    pd.DataFrame([[
                        *np.array(image).astype(int).flatten(),
                        *one_hot_encoder(letter, alphabet),
                        letter
                    ]], columns=df.columns),
                    df
                ],
                ignore_index=True
            )

    df[features + one_hot_encode_labels] = df[features + one_hot_encode_labels].astype('int')
    return df[features], df[one_hot_encode_labels]


def one_hot_encoder(label, labels):
    labels = np.array(labels)
    index = np.where(labels == label)[0][0]
    one_hot_vector = np.zeros(len(labels), dtype=int)
    one_hot_vector[index] = 1
    return one_hot_vector
