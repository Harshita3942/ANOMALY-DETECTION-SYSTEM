import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

class AnomalyDetection:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.model = None

    def train_model(self):
        # Initialize the IsolationForest model
        self.model = IsolationForest(contamination=0.1)
        self.model.fit(self.data)

    def detect_anomalies(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        # Predict anomalies (-1 for anomalies, 1 for normal)
        anomalies = self.model.predict(self.data)
        return anomalies

    def visualize_results(self, anomalies):
        plt.figure(figsize=(10,6))
        plt.scatter(range(len(self.data)), self.data, c=anomalies, cmap='coolwarm')
        plt.xlabel('Data Index')
        plt.ylabel('Feature Value')
        plt.title('Anomaly Detection')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generating sample data
    data = pd.DataFrame(np.random.randn(100, 1), columns=['Feature'])
    # Add some anomalies
    data.iloc[5] = 10
    data.iloc[50] = -10
    
    # Initialize and train the model
    anomaly_detector = AnomalyDetection(data)
    anomaly_detector.train_model()
    
    # Detect anomalies
    anomalies = anomaly_detector.detect_anomalies()

    # Visualize results
    anomaly_detector.visualize_results(anomalies)
