import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay

def evaluate_and_save_metrics(data_path, model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and test data
    pipeline = joblib.load(os.path.join(model_dir, 'crop_model.pkl'))
    X_test, y_test = joblib.load(os.path.join(model_dir, 'test_data.pkl'))
    
    # Predict and evaluate
    predictions = pipeline.predict(X_test)
    bacc = balanced_accuracy_score(y_test, predictions)
    
    # Save metrics to text
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Balanced Accuracy: {bacc:.3f}\n")
    
    # Generate and save confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, xticks_rotation=45)
    plt.title(f'Confusion Matrix (BAcc: {bacc:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()