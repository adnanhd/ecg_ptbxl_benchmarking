import torch
import logging
import torchvision.models as models
from models.base_model import ClassificationModel


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.y.__len__()

    def __getitem__(self, index):
        return self.x[index], self.y[index]


logging.basicConfig(filemode="deneme.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ResNet18(ClassificationModel):
    def __init__(self,
                 name: str,
                 n_classes: int,
                 sampling_frequency,
                 outputfolder,
                 input_shape,
                 out_dimensions: int = 3,
                 model_dtype: torch.dtype = None,
                 model_device: torch.device = None):
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape
        self.model = models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.model.fc = torch.nn.Linear(2048, out_dimensions)
        if model_dtype is not None:
            self.model.to(dtype=model_dtype)
        if model_device is not None:
            self.model.to(device=model_device)
        params = next(self.model.parameters())
        self.device = params.device
        self.dtype = params.dtype
        logger.debug(f"input_shape {input_shape}")
        logger.debug(f"model device {self.device}")
        logger.debug(f"model dtype {self.dtype}")
        logger.debug(f"n_classes {n_classes}")
        self.n_epoch = 30
        self.batch_size = 128
        self.lr = 5e-4
        self.valid_epochs = 5

    def create_features(self, input):
        input = torch.from_numpy(input).to(device=self.device)
        logger.debug(f"input dtype {input.dtype}")
        return input[:, :, :2].transpose(2, 1).unsqueeze(-1)

    def create_labels(self, target):
        target = torch.from_numpy(target).to(device=self.device)
        v0, _ = target[:, :13].max(1, keepdims=True)
        v1, _ = target[:, 13:18].max(1, keepdims=True)
        v2, _ = target[:, 18:32].max(1, keepdims=True)
        # v3, _ = target[:, 32:43].max(1, keepdims=True)
        # v4, _ = target[:, 43].max(1, keepdims=True)
        result = torch.cat([v0, v1, v2], dim=1)
        # inverse of one hot encoding
        _, indices = result.max(1, keepdims=False)
        return indices

    def fit(self, X_train, y_train, X_val, y_val):
        logger.debug(f"x_train {X_train.shape}")
        logger.debug(f"x_val {X_val.shape}")
        logger.debug(f"y_train {y_train.shape}")
        logger.debug(f"y_val {y_val.shape}")

        train_dataset = Dataset(self.create_features(X_train),
                                self.create_labels(y_train))
        valid_dataset = Dataset(self.create_features(X_val),
                                self.create_labels(y_val))
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=len(valid_dataset))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.n_epoch + 1):
            total_loss = 0
            count_loss = 0
            for x, y in torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True):
                pred = self.model(x)
                logger.debug(
                    f"train shape {x.shape} prediction shape {pred.shape} output shape {y.shape}")
                loss = torch.nn.functional.cross_entropy(input=pred, target=y)
                total_loss += loss.item()
                count_loss += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.info(
                f"Epoch {epoch} of {self.n_epoch}: Loss {total_loss / count_loss}")
            if epoch % self.valid_epochs == 0:
                with torch.no_grad():
                    for x, y in valid_dataloader:
                        pred = self.model(x)
                        loss = torch.nn.functional.cross_entropy(
                            input=pred, target=y)

    def predict(self, X):
        return self.model(self.create_features(X))
