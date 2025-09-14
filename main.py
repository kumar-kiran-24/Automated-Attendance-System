from src.logger import logging
from src.exception import CustomException   

from src.components.trained_models.cnn_model import CNNModel
import sys
from dataclasses import dataclass

from flask import Flask, request

