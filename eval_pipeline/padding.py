import numpy as np

from netcal.metrics import ECE
n_bins = 10
ece = ECE(n_bins)
confidences = np.array([[0.1, 0.9], [0.3, 0.3, 0.4]])
print(confidences.shape)
answers = np.array([1, 0])
ece_output = ece.measure(confidences, answers)
print(ece_output)