import torch
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMG_SIZE = 64
IMG_CHANNEL = 1
NOISE_DIM = 100
NUM_EPOCH = 100
DISC_FEATURES = 64
GEN_FEATURES = 64
DATA_DIR = "D:\AI\DCgan\datasets\disgust"
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMG_CHANNEL)], [0.5 for _ in range(IMG_CHANNEL)]
        )
    ]
)
NUM_WORKERS = 0
SAVE_PATH = "generated"
MLFLOW_EXP = 'dcgan'
MLFLOW_SOURCE = './mlruns'
