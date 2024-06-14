# Protein Sequence Classification Project

This repository contains the code for classifying protein sequences using an RNN model.

## Project Structure

- `protein_sequence_classification.py`: The main script that loads data, preprocesses it, defines the RNN model, and trains it.

## Requirements

To run this project, you need to have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- TensorFlow
- Scikit-learn

You can install the required Python packages using:

pip install numpy pandas tensorflow scikit-learn

## Dataset
The dataset should be a CSV file with columns 'sequence' and 'label'. Update the file path in the script accordingly.

## Usage
1. Load Data:

The load_data function loads the CSV file into a Pandas DataFrame.

data_file = 'path/to/protein_data.csv'
sequences, labels = load_data(data_file)

2. Preprocess Data:

The sequences are tokenized and padded, and the labels are encoded.

sequences, labels, tokenizer, label_encoder = preprocess_data(sequences, labels)

3. Train-Test Split:

The data is split into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

4. Build Model:

An RNN model is built using TensorFlow's Keras API.

model = build_model(input_length=X_train.shape[1], vocab_size=len(tokenizer.word_index) + 1)

5. Train Model:

The model is trained on the training data.

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

6. Save Model:

The trained model, tokenizer, and label encoder are saved.

model.save('protein_sequence_classification_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
## Contributing
If you have any suggestions or improvements, feel free to open an issue or create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/JaCar-868/Disease-Progression/blob/main/LICENSE) file for details.
