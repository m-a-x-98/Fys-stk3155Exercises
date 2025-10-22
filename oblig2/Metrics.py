import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    DetCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve
)
import matplotlib.pyplot as plt

class ClassificationMetrics:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.pred = model.predictClasses(x_test)
        self.y_test = y_test

    def oneFunctionToShowThemAll(self, binary):
        self.ConfusionMatrix(show=False)
        if (binary): self.ROCcurve(show=False)
        if (binary): self.PrecisionRecall(show=False)
        if (binary): self.CumulativeGain(show=False)
        self.PrecisionRecallROC(show = False)
        plt.show()

    def ConfusionMatrix(self, show : bool = True):     
        ConfusionMatrixDisplay.from_predictions(
            self.pred, self.y_test,
            normalize='true',
            cmap='Blues'
        )
        plt.title("Normalized Confusion Matrix")
        if (show): plt.show()

    def ROCcurve(self, show : bool = True):
        RocCurveDisplay.from_predictions(self.pred, self.y_test)
        plt.title("ROC Curve")
        if (show): plt.show()

    def PrecisionRecall(self, show : bool = True):
        PrecisionRecallDisplay.from_predictions(self.pred, self.y_test)
        plt.title("Precision-Recall Curve")
        if (show): plt.show()

    def CumulativeGain(self, show : bool = True):
        fpr, tpr, _ = roc_curve(self.y_test, self.model.predict(self.x_test)[:, 1])
        plt.plot(np.arange(len(tpr)) / len(tpr), tpr, label='Cumulative Gain')
        plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlabel('Proportion of Sample')
        plt.ylabel('Proportion of Positives Captured')
        plt.title('Cumulative Gain Chart')
        plt.legend()
        if (show): plt.show()


    def PrecisionRecallROC(self, show : bool = True):
        nClasses = 3

        threshold = 0
        all_precisions = [[] for _ in range(nClasses)]
        all_recalls = [[] for _ in range(nClasses)]
        all_tprs = [[] for _ in range(nClasses)]
        all_fprs = [[] for _ in range(nClasses)]
        while threshold < 1:
            for i in range(nClasses):
                probs = self.model.predict(self.x_test).copy()
                probs[(probs < threshold)] = 0 # set all values less than threshold to 0
                pred_classes = np.where(probs.max(axis=1) == 0, -1, np.argmax(probs, axis=1))
                y_true = (self.y_test == i).astype(int)
                y_pred = (pred_classes == i).astype(int)
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))

                tpr = tp / (tp + fn + 1e-9)
                fpr = fp / (fp + tn + 1e-9)
                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)
                
                all_tprs[i].append(tpr)
                all_fprs[i].append(fpr)
                all_precisions[i].append(precision)
                all_recalls[i].append(recall)
            threshold += 0.1

        # --- Plot Precision-Recall Curve ---
        plt.figure(figsize=(8, 5))
        for i in range(nClasses):
            plt.plot(all_recalls[i], all_precisions[i], label=f'Class {i}')
        plt.plot(np.mean(all_recalls, axis = 0), np.mean(all_precisions, axis = 0), label=f"Average")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid()
        if (show): plt.show()

        # --- Plot ROC Curve ---
        plt.figure(figsize=(8, 5))
        for i in range(nClasses):
            plt.plot(all_fprs[i], all_tprs[i], label=f'Class {i}')
        plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid()
        if (show): plt.show()
