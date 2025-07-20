# Predictive Maintenance Optimization for Industrial Machinery

This project provides an advanced solution for predicting the Remaining Useful Life (RUL) of industrial machinery using deep learning. By forecasting equipment failure, this tool enables proactive maintenance scheduling, aiming to significantly reduce operational disruptions and costs associated with unplanned downtime.

The core of this project is a Deep Neural Network built with **PyTorch**, featuring a custom architecture designed for time-series regression. The entire pipeline, from data preprocessing to model evaluation, is included.

##  Key Features

- **Advanced Regression Model**: A custom-built Deep Neural Network tailored for predicting RUL with high accuracy.
- **NASA CMAPSS Dataset**: Utilizes the well-established NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset for robust training and evaluation.
- **Custom Loss Function**: Implements a unique loss function that penalizes late predictions (over-predictions) more heavily than early ones, aligning the model's performance with real-world maintenance logistics.
- **Modern Techniques**: Incorporates **Leaky ReLU** activation, **weight normalization**, and **early stopping** to enhance model performance and prevent overfitting.
- **Comprehensive Evaluation**: Model performance is rigorously assessed using Mean Absolute Error (MAE), Mean Squared Error (MSE), and the R² Score.
- **Rich Visualizations**: Automatically generates and saves plots illustrating the RUL degradation over operational cycles for clear performance analysis.

## 📈 Performance

The model achieves exceptional predictive accuracy, demonstrating its reliability in tracking equipment degradation.

- **Mean Absolute Error (MAE)**: 0.94
- **Mean Squared Error (MSE)**: 1.39
- **R² Score**: 0.9992

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and pip installed.

### Installation & Execution

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Run the main script to train and evaluate the model:**
    ```sh
    python main.py
    ```

4.  **View the results:**
    Check the `plots/` directory to see the generated visualizations of the RUL predictions.
