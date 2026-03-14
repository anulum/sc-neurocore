# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any
import numpy as np

from ..layers.sc_conv_layer import SCConv2DLayer
from ..layers.vectorized_layer import VectorizedSCLayer


class SCDigitClassifier:
    """
    Pre-configured SC Network for MNIST-like Digit Classification.
    Uses: Conv Layer -> Vectorized Dense Layer
    """

    def __init__(self) -> None:
        # 1. Convolutional Front-End (Feature Extraction)
        # Input: 28x28, 1 Channel
        self.conv = SCConv2DLayer(
            in_channels=1, out_channels=4, kernel_size=3, stride=2, length=256
        )
        # Output map size: (28-3)/2 + 1 = 13x13 -> 4x13x13 = 676 features

        # 2. Dense Classifier
        self.dense = VectorizedSCLayer(n_inputs=676, n_neurons=10, length=1024)  # 10 Digits

    def forward(self, image: np.ndarray[Any, Any]) -> int:
        """
        Classify a 28x28 image.
        """
        # Ensure correct shape (1, 28, 28)
        if image.ndim == 2:
            image = image[None, :, :]

        # 1. Conv
        features = self.conv.forward(image)

        # Flatten
        flat_features = features.flatten()

        # 2. Dense
        # Vectorized layer expects list/array of floats as probabilities
        # We need to map the conv output (accumulated bit counts) to probabilities [0,1]
        # Conv output is roughly sum of bits. Max bits = kernel_size^2 * length?
        # Let's normalize assuming max overlap
        norm_factor = (3 * 3) * 256
        flat_probs = flat_features / norm_factor
        flat_probs = np.clip(flat_probs, 0, 1)

        outputs = self.dense.forward(flat_probs)

        # Argmax
        return int(np.argmax(outputs))


class SCKeywordSpotter:
    """
    Audio Keyword Spotter (e.g., "Yes"/"No").
    Uses: Recurrent / Dense Layer
    """

    def __init__(self, n_keywords=2) -> None:
        # Simple Dense for demo (RNN requires sequence logic implementation)
        self.classifier = VectorizedSCLayer(n_inputs=16, n_neurons=n_keywords)

    def predict(self, mfcc_features: np.ndarray[Any, Any]) -> int:
        return int(np.argmax(self.classifier.forward(mfcc_features)))
