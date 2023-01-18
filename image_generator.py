import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os


def save_plot(examples, n):
    examples = (examples + 1)/2.0
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(examples[i])
    file_name = f"samples/fake_image.png"
    try:
        plt.savefig(file_name)
    except:
        os.mkdir("/samples")
        plt.savefig(file_name)
    plt.close()


if __name__ == "__main__":
    model = load_model("saved_model\g_model.h5")
    n_samples = 25
    latent_dim = 128
    latent_points = np.random.normal(size=(n_samples, latent_dim))
    examples = model.predict(latent_points)
    save_plot(examples, 5)