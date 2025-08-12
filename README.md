# Deep Learning Regression Models with Keras

In this project, I explored building and training neural network models for regression tasks using Keras and TensorFlow. I implemented models on the California Housing dataset to predict median house values and on a synthetic dataset to demonstrate a simple linear regression with added noise. The primary focus here is on showcasing methods, techniques, and workflows in deep learning—such as different Keras APIs, data preprocessing, callbacks, and hyperparameter tuning—rather than solely optimizing for the lowest error percentages or highest metrics. Some models intentionally use basic or experimental configurations to illustrate core concepts, mechanics, and learning processes, highlighting my understanding of the underlying principles over peak performance.

## Problem Statement and Goal of Project

The primary goal was to predict continuous values using deep learning regression models, emphasizing the demonstration of methodologies.

- For the California Housing dataset: Predict median house prices based on features like longitude, latitude, housing median age, total rooms, and others. This serves to illustrate handling real-world tabular data for regression while exploring various model architectures.
- For the synthetic dataset: Model a linear relationship with noise (Y = X @ weights + bias + noise) to showcase fundamental regression mechanics, data generation, and evaluation in a controlled environment.

These tasks allowed me to practice end-to-end workflows, from data splitting and scaling to model evaluation, with an emphasis on experimenting with Keras APIs and techniques to deepen my understanding of methods rather than focusing on error rates.

## Solution Approach

I followed a structured approach for both notebooks, prioritizing the display of methods:

1. **Data Preparation**: Loaded or generated data, split into train/test/validation sets using `train_test_split`, and applied `StandardScaler` for feature normalization.
2. **Model Building**:
   - Used Sequential API for a multi-layer perceptron with Dense layers, BatchNormalization, and Dropout to demonstrate standard regression setups.
   - Demonstrated Functional API for more complex architectures, including multiple inputs/outputs and concatenation layers, to show flexible model design.
   - In the synthetic data notebook, incorporated an Input layer explicitly to highlight input handling.
3. **Compilation and Training**: Compiled models with optimizers (Adam or SGD), MSE loss, and MAE metric. Trained with validation data and callbacks like EarlyStopping, ModelCheckpoint, and TensorBoard to illustrate efficient training and monitoring techniques.
4. **Hyperparameter Tuning**: Wrapped a Keras model in `KerasRegressor` and used `GridSearchCV` to tune parameters like learning rate, number of hidden layers, and neurons, showcasing integration with scikit-learn for optimization workflows.
5. **Evaluation**: Assessed models using MSE, MAE, and RMSE on test data, and printed predictions for comparison. Low metrics in some cases are intentional to demonstrate learning about potential issues like NaNs or basic setups.
6. **Exploration**: Extracted and printed model weights/biases to inspect learned parameters, and saved models for reuse, emphasizing interpretability and reusability.

This implementation helped me showcase various methods, such as API variations, overfitting mitigation, and hyperparameter search, with a focus on educational value over minimizing error percentages.

## Technologies & Libraries

- **Core Frameworks**: TensorFlow (for backend), Keras (for model building).
- **Data Handling**: scikit-learn (datasets, train_test_split, StandardScaler, GridSearchCV, KerasRegressor).
- **Utilities**: NumPy (data generation/manipulation), Matplotlib (imported but not used in visible code), datetime (for TensorBoard logs).
- **Environment**: Python 3.x (inferred from code; one notebook uses 3.9.21, the other 3.8.20).
- **Callbacks**: ModelCheckpoint, EarlyStopping, TensorBoard.

## Description about Dataset

- **California Housing Dataset** (from `sklearn.datasets.fetch_california_housing`): Contains 20,640 samples with 8 features (e.g., MedInc, HouseAge, AveRooms). Target is median house value (in hundreds of thousands). Split into train (15,480), validation (3,870), and test (5,160) samples.
- **Synthetic Dataset**: Generated 10,000 samples with 3 input features (random values between 0 and 1). Target computed as a linear combination: Y = X @ [2.0, -3.5, 1.0] + 5.0 + Gaussian noise (mean=0, std=0.1). Split into train (8,000), validation (2,000 from train split), and test (2,000).

## Installation & Execution Guide

1. **Prerequisites**: Install Python 3.x and required libraries via pip:
   ```
   pip install tensorflow scikit-learn numpy matplotlib tensorboard
   ```
2. **Run Notebooks**:
   - Open `regression.ipynb` or `random_regression.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute cells sequentially. For TensorBoard in `random_regression.ipynb`, run `%load_ext tensorboard` and `tensorboard --logdir logs/fit` to view training logs.
3. **Model Saving/Loading**: Models are saved as HDF5 (e.g., `model_regresion_p1.h5`) or TensorFlow format (`model_regresion_p1`). Load with `keras.models.load_model()`.

## Key Results / Performance

Performance metrics are presented to illustrate the outcomes of the methods used, without primary focus on minimizing errors:

- **California Housing Model (Sequential API)**: After 30 epochs, validation MSE ~0.3463, MAE ~0.4271. Demonstrates stable training with Dropout and BatchNormalization.
- **Functional API Model**: Trained with multiple outputs; validation losses ~0.3595 (main) and ~0.4215 (helper). Explored input splitting and weighted losses.
- **GridSearchCV Tuning**: Best params: learning_rate=0.01, hidden_layers=5, neurons=50; best R² score ~0.6852. Some configurations yielded NaN, intentionally highlighting learning about edge cases in experimental setups.
- **Synthetic Model**: Test MSE ~0.0104, MAE ~0.0815, RMSE ~0.1019. Predictions closely match actuals (e.g., Actual: 4.7478, Predicted: 4.9148), showing effective application of techniques despite noise. Any lower metrics serve to demonstrate basic approaches for educational purposes.

## Screenshots / Sample Outputs

Sample training output from Sequential model (California Housing):
```
Epoch 1/30
363/363 [==============================] - 3s 5ms/step - loss: 2.2364 - mae: 1.1367 - val_loss: 0.6114 - val_mae: 0.5866
...
Epoch 30/30
363/363 [==============================] - 3s 7ms/step - loss: 0.3858 - mae: 0.4471 - val_loss: 0.3463 - val_mae: 0.4271
```

Sample predictions from synthetic model:
```
Actual: [4.7478204], Predicted: [4.91477]
Actual: [6.3260946], Predicted: [6.4038734]
```

GridSearchCV best params:
```
{'model__learning_rate': 0.01, 'model__number_of_hidden_layers': 5, 'model__number_of_neurons': 50}
```

(Plots like training history are mentioned via Matplotlib imports but not executed in the provided code.)

## Additional Learnings / Reflections

Through this project, I gained hands-on experience with Keras APIs: Sequential for straightforward models, Functional for flexible architectures with multiple inputs/outputs. I learned to integrate scikit-learn for tuning and preprocessing, handle issues like NaNs in experimental configs, and use callbacks to manage training efficiently while monitoring via TensorBoard. The synthetic dataset helped benchmark linear models, reinforcing concepts like weight initialization and noise impact. The emphasis was on demonstrating methods and workflows, which solidified my skills in regression techniques and prepared me for applying these in more complex AI scenarios, rather than chasing minimal error rates.

## 👤 Author

## Mehran Asgari
## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).
## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

---

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*