import io
import os
from functools import lru_cache
from os import path
from typing import Callable, Literal

import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageEnhance
from PIL.Image import Image as TImage
from polars import DataFrame, read_csv
from torch import nn
from torch._prims_common import DeviceLikeType
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchinfo import (
    ModelStatistics,
    summary,  # type: ignore
)
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Lambda,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

from src.clients.schemas.model.model import BirdSpeciePrediction, Prediction

Datasets = Literal["train", "test", "valid"]
Devices = Literal["cpu", "cuda"]


class DEITTinyModel(nn.Module):
    def __init__(self):
        super().__init__()  # type: ignore
        self.model = torch.hub.load(  # type: ignore
            "facebookresearch/deit:main",
            "deit_tiny_patch16_224",
            pretrained=True,
        )


class ModelService:
    def __init__(
        self,
        train_data_path: str = "train/",
        test_data_path: str = "test/",
        valid_data_path: str = "valid/",
        dataset_classes_path: str = "birds.csv",
        model_output_path: str = "2024-03-03-model.pth",
    ) -> None:
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_dir: str = os.path.join(self.dir_path, "data")
        self.train_data_path: str = train_data_path
        self.test_data_path: str = test_data_path
        self.valid_data_path: str = valid_data_path
        self.dataset_classes_path: str = dataset_classes_path
        self.model_output_path: str = model_output_path

        self.batch_size: int = 64

        self.train_transform = Compose(
            [
                Resize(
                    (224, 224),
                    InterpolationMode.LANCZOS,
                ),
                RandomHorizontalFlip(),
                Lambda(self.enhance),
                ToTensor(),
            ]
        )

        self.valid_transform = Compose(
            [
                Resize(
                    (224, 224),
                    InterpolationMode.LANCZOS,
                ),
                ToTensor(),
            ]
        )

        self.train_phases: list[Datasets] = ["train", "valid"]

        self.class_count: int | None = None

    @lru_cache()
    def get_classes(
        self,
        column_name: str = "labels",
    ) -> list[str]:
        assert self.data_dir is not None, "`data_dir` is required"
        assert (
            self.dataset_classes_path is not None
        ), "`dataset_classes_path` is required"

        classes_path = path.join(self.data_dir, self.dataset_classes_path)
        df_classes: DataFrame = read_csv(classes_path)

        classes: list[str] = df_classes[column_name].to_list()

        return classes

    def get_dataset(self, set: Datasets) -> ImageFolder:
        assert self.data_dir is not None, "`data_dir` is required"

        dataset_dict: dict[str, str] = dict(
            train=path.join(self.data_dir, self.train_data_path),
            test=path.join(self.data_dir, self.test_data_path),
            valid=path.join(self.data_dir, self.valid_data_path),
        )

        transform = self.valid_transform
        if set == "train":
            transform = self.train_transform

        root = dataset_dict[set]

        dataset = ImageFolder(
            root=root,
            transform=transform,
        )

        self.class_count = len(dataset.classes)
        logger.info(f"Class count: {self.class_count}")

        return dataset

    def get_dataloader(self, set: Datasets) -> DataLoader[bytes]:
        dataset = self.get_dataset(set=set)

        dataloader: DataLoader[bytes] = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        logger.info(f"Dataloader '{set}' length: {len(dataloader)}")

        return dataloader

    def get_model(self) -> DEITTinyModel:
        model = DEITTinyModel().model
        _device = self.get_device()

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.head.in_features

        self.get_dataloader(set="train")

        assert (
            self.class_count is not None
        ), "Class count is `None`, need to call `get_dataloader`"

        model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, (self.class_count)),
        )
        model = model.to(_device)

        model_summary: ModelStatistics = summary(model, input_size=(1, 3, 224, 224))
        logger.info(f"Model info: {model_summary}")

        return model

    def get_device(
        self,
    ) -> DeviceLikeType:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _device

    def color_enhance(
        self,
        image: TImage,
        factor: float,
    ) -> TImage:
        return ImageEnhance.Color(image).enhance(factor)  # type: ignore

    def contrast_enhance(
        self,
        image: TImage,
        factor: float,
    ) -> TImage:
        return ImageEnhance.Contrast(image).enhance(factor)  # type: ignore

    def brighness_enhance(
        self,
        image: TImage,
        factor: float,
    ) -> TImage:
        return ImageEnhance.Brightness(image).enhance(factor)  # type: ignore

    def sharpness_enhance(
        self,
        image: TImage,
        factor: float,
    ) -> TImage:
        return ImageEnhance.Sharpness(image).enhance(factor)  # type: ignore

    def enhance(
        self,
        image: TImage,
    ) -> TImage:
        order = [0, 1, 2, 3]
        np.random.shuffle(order)

        enhancers: list[
            tuple[
                Callable[[TImage, float], TImage],
                Callable[[], float],
            ]
        ] = [
            (self.color_enhance, lambda: np.random.normal(1.0, 0.3)),
            (self.contrast_enhance, lambda: np.random.normal(1.0, 1.0)),
            (self.brighness_enhance, lambda: np.random.normal(1.0, 1.0)),
            (self.sharpness_enhance, lambda: np.random.normal(1.0, 0.3)),
        ]
        for i in order:
            enhancer, factor = enhancers[i]
            image = enhancer(image, factor())  # type: ignore

        assert isinstance(image, TImage)

        return image

    def train_model(
        self,
        num_epochs: int = 10,
        save_model: bool = True,
    ) -> DEITTinyModel:
        model: DEITTinyModel = self.get_model()
        _device = self.get_device()

        train_loader = self.get_dataloader(set="train")
        valid_loader = self.get_dataloader(set="valid")
        train_data = self.get_dataset(set="train")
        valid_data = self.get_dataset(set="valid")

        criterion = nn.CrossEntropyLoss(
            label_smoothing=0.11,
        )
        optimizer = AdamW(
            model.parameters(),
            lr=0.0001,
        )

        history: list[list[float]] = list()

        for epoch in range(num_epochs):
            model.train()

            train_loss: float = 0.0
            train_acc: float = 0.0

            valid_loss: float = 0.0
            valid_acc: float = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(_device), labels.to(_device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                if i % 100 == 0:
                    print("Step-{},Loss-{}".format(i, loss.item()))
                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)

                # Compute the accuracy
                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))  # type: ignore

                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)

            # Validation - No gradient tracking needed
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for _, (inputs, labels) in enumerate(valid_loader):
                    inputs = inputs.to(_device)
                    labels = labels.to(_device)

                    # Forward pass - compute outputs on input data using the model
                    outputs = model(inputs)

                    # Compute loss
                    loss = criterion(outputs, labels)

                    # Compute the total loss for the batch and add it to valid_loss
                    valid_loss += loss.item() * inputs.size(0)

                    # Calculate validation accuracy
                    _, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    # Convert correct_counts to float and then compute the mean
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))  # type: ignore

                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)

            # Find average training loss and training accuracy
            avg_train_loss = train_loss / len(train_data)
            avg_train_acc = train_acc / len(train_data)

            # Find average training loss and training accuracy
            avg_valid_loss = valid_loss / len(valid_data)
            avg_valid_acc = valid_acc / len(valid_data)

            history.append(
                [avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc]
            )
            print(history)

            print(
                "Epoch : {}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%".format(
                    epoch + 1,
                    avg_train_loss,
                    avg_train_acc * 100,
                    avg_valid_loss,
                    avg_valid_acc * 100,
                )
            )

        if save_model is True:
            output_path = os.path.join(
                self.dir_path,
                self.model_output_path,
            )
            torch.save(  # type: ignore
                model.state_dict(),
                output_path,
            )

        return model

    def test_model(
        self,
    ) -> None:
        model: DEITTinyModel = self.get_model()
        _device = self.get_device()

        test_loader = self.get_dataloader(set="test")
        test_data = self.get_dataset(set="test")

        criterion = nn.CrossEntropyLoss(
            label_smoothing=0.11,
        )

        history: list[list[float]] = list()

        model.eval()

        valid_loss: float = 0.0
        valid_acc: float = 0.0

        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            # Validation loop
            for _, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(_device)
                labels = labels.to(_device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))  # type: ignore

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

        avg_test_loss = valid_loss / len(test_data)
        avg_test_acc = valid_acc / len(test_data)

        history.append([avg_test_loss, avg_test_acc])
        print(history)

        print(
            "Test: Loss: {:.4f}, Accuracy: {:.4f}%".format(
                avg_test_loss,
                avg_test_acc * 100,
            )
        )

    def load_model(
        self,
    ) -> DEITTinyModel:
        _device = self.get_device()

        model = self.get_model()
        model_path = path.join(self.dir_path, self.model_output_path)
        model.load_state_dict(torch.load(model_path))  # type: ignore
        model.to(_device)

        return model

    def process_image(
        self,
        image: bytes,
    ) -> torch.Tensor:
        pil_image: Image = Image.open(io.BytesIO(image)).convert("RGB")  # type: ignore

        transform = Compose([Resize((224, 224)), ToTensor()])
        img_normalized = transform(pil_image).float()  # type: ignore

        if torch.cuda.is_available():
            test_image_tensor = img_normalized.view(1, 3, 224, 224).cuda()  # type: ignore
        else:
            test_image_tensor = img_normalized.view(1, 3, 224, 224)  # type: ignore

        assert isinstance(test_image_tensor, torch.Tensor)

        return test_image_tensor

    def predict(
        self,
        image: bytes,
        top_k: int,
    ) -> Prediction:
        _device: DeviceLikeType = self.get_device()
        model: DEITTinyModel = self.load_model()
        processed_image_tensor = self.process_image(
            image=image,
        ).to(_device)

        train_data = self.get_dataset(set="train")
        idx_to_class = {v: k for k, v in train_data.class_to_idx.items()}

        prediction = Prediction(
            specie=None,
            top_k=list(),
        )

        with torch.no_grad():
            model.eval()
            out = model(processed_image_tensor)
            prob = torch.exp(out)
            prob_, class_ = prob.topk(3, dim=1)
            class_ = class_.cpu().numpy()

            for i in range(top_k):
                specie_name: str = idx_to_class[class_[0][i]]
                score: float = prob_.cpu().numpy()[0][i]
                specie = BirdSpeciePrediction(
                    specie_name=specie_name,
                    score=score,
                )
                if i == 0:
                    prediction.specie = specie
                prediction.top_k.append(specie)

        return prediction
