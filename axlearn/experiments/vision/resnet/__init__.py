# Copyright © 2023 Apple Inc.

"""ResNet trainer module."""
import pytest

from . import imagenet_trainer
from .common import learner_config, model_config

pytestmark = pytest.mark.vision