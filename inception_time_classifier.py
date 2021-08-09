import logging
from typing import Iterable, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader

from .subcomponents import InceptionBlock, Flatten


logger = logging.getLogger(__name__)


class TSDataset(Dataset):
    def __init__(self, X:torch.Tensor, y:torch.Tensor):
        self.X = X
        self.y = y

    def __getitem__(self, instance_index) -> Tuple[torch.Tensor]:
        return self.X[instance_index, :, :], self.y[instance_index]

    def __len__(self):
        return self.y.shape[0]


def on_trained_epoch(clf, i_epoch:int, y_true_epoch:torch.Tensor, y_pred_epoch:torch.tensor):
    epoch_stats = {
        'epoch': i_epoch + 1,
        'loss': {
            'name': clf.criterion._get_name(), 'value': clf.criterion(y_pred_epoch, y_true_epoch)
        }
    }
    if  i_epoch in [0, 1, clf.n_epochs] or (i_epoch + 1) % 5 == 0:
        with torch.no_grad():
            print(
                f'epoch={epoch_stats["epoch"]}/{clf.n_epochs} \t'
                f'{epoch_stats["loss"]["name"]}={epoch_stats["loss"]["value"]:.3f}'
            )
    return epoch_stats


def on_trained_model(clf, y_true:torch.Tensor, y_pred:torch.Tensor):
    print(f'finished training!')
    return None


class InceptionTimeClassifier(ClassifierMixin):
    def __init__(
            self, n_epochs:int=100, lr:float=1e-3, batch_size:int=32,
            optimizer_class=torch.optim.Adam, loss_class=nn.CrossEntropyLoss,
            dataloader_class=torch.utils.data.DataLoader, dataset_class=TSDataset,
            on_trained_epoch:Callable=on_trained_epoch, on_trained_model:Callable=on_trained_model,
            device:str='cuda'
    ):
        """
        dataset_class: must inherit from pytorch.utils.data, and its contstructor must have two inputs: X, y.
        on_trained_epoch: function with inputs = (classifier, i_epoch, n_epochs, y_true_epoch, y_pred_epoch)
            that outputs the epoch statistics that will be stored in the model attribute epochs_statistics
            called when finished training an epoch
        on_trained_model: function with inputs = (classifier, y_true, y_pred) and outputs the train statistics
            that will be stored in the model attribute train_statistics
            called when finished training all epochs
        """
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.loss_class = loss_class
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dataloader_class = dataloader_class
        self.dataset_class = dataset_class
        self.on_trained_epoch = on_trained_epoch
        self.on_trained_model = on_trained_model
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                logger.warning('No CUDA available: using cpu')
                self.device = torch.device("cpu")
        # attributes assigned during fit
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.epochs_stats = []
        self.training_stats = None

    def _train_epoch(self, i_epoch:int, dataloader:DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        y_true_epoch, y_pred_epoch = torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        for i_batch, (X_batch, y_true_batch) in enumerate(iter(dataloader)):
            y_true_batch, y_pred_batch  = self._train_batch(i_batch, X_batch, y_true_batch)
            y_true_epoch, y_pred_epoch = torch.cat((y_true_epoch, y_true_batch)), torch.cat((y_pred_epoch, y_pred_batch))
        self.epochs_stats.append(self.on_trained_epoch(self, i_epoch, y_true_epoch, y_pred_epoch))
        return y_true_epoch, y_pred_epoch

    def _train_batch(self, i_batch:int, X_batch:torch.Tensor, y_true_batch:torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        self.optimizer.zero_grad()
        y_pred_batch = self.model(X_batch)
        loss = self.criterion(y_pred_batch, y_true_batch)
        loss.backward()
        self.optimizer.step()
        return y_true_batch, y_pred_batch

    def fit(self, X:np.ndarray, y:np.ndarray or pd.Series):
        """
        X: 3d numpy array with axes = (sample, feature, time)
        y: 2d numpy array with axes (sample, class) or 1d in case of binary classification
        """
        n_samples = X.shape[0]
        n_input_channels = X.shape[1]
        n_times = X.shape[2]
        assert X.ndim == 3 and y.ndim in [1, 2] and X.shape[0] == y.shape[0]
        X, y = torch.tensor(X, device=self.device).float(), torch.tensor(y, device=self.device).float()
        if y.ndim == 1:  # binary classification where user passed just the probabilty of the positive class
            n_classes = 2
            with torch.no_grad():
                y = torch.stack((y, y), dim=1)
                y[:, 0] = 1 - y[:, 1]
        else:
            n_classes = y.shape[1] # TODO: I'm not conviced: probability should sum to 1 so one class is redundant
        if n_classes == 2 and self.loss_class == nn.CrossEntropyLoss:
            self.loss_class = nn.BCEWithLogitsLoss
        dataset = self.dataset_class(X, y)
        dataloader = self.dataloader_class(dataset=dataset, batch_size=self.batch_size, shuffle=False)

        self.model = nn.Sequential(
                    InceptionBlock(
                        in_channels=n_input_channels,
                        n_filters=32,
                        kernel_sizes=[5, 11, 23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    InceptionBlock(
                        in_channels=32*4,
                        n_filters=32,
                        kernel_sizes=[5, 11, 23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    nn.AdaptiveAvgPool1d(output_size=1),
                    Flatten(out_features=32*4*1),
                    nn.Linear(in_features=4*32*1, out_features=n_classes)
        )

        self.model = self.model.to(self.device)
        self.criterion = self.loss_class()
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)

        y_true, y_pred = [torch.empty([self.n_epochs, n_samples, n_classes], device=self.device)] * 2
        self.model.train()
        for epoch in range(self.n_epochs):
            try:
                y_true[epoch], y_pred[epoch] = self._train_epoch(epoch, dataloader)
            except KeyboardInterrupt:
                break
        self.training_stats = self.on_trained_model(self, y_true, y_pred)
        return self

    def predict(self, X:np.ndarray or pd.DataFrame or torch.Tensor) -> np.ndarray:
        y_pred = np.argmax(self.predict_proba(X), axis=1)
        return y_pred

    def predict_proba(self, X:np.ndarray or pd.DataFrame or torch.Tensor) -> np.ndarray:
        assert X.ndim >= 2
        X = torch.tensor(X, device=self.device).float()
        with torch.no_grad():
            self.model.eval()
            proba_pred = (self.model(X)).detach().cpu().numpy()  # TODO: it's not a probability but values (-inf, inf)
        return proba_pred
