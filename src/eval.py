from typing import List, Tuple

import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from PIL import Image
import torch
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
import numpy as np

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.dlib_datamodule import TransformDataset  # noqa: E402
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    metric_dict = trainer.callback_metrics

    # for predictions use trainer.predict(...)
    log.info("Starting predictions!")
    # predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    # print("predictions:", len(predictions), type(predictions[0]), predictions[0].shape)
    # for pred, (bx, by) in zip(predictions, datamodule.predict_dataloader()):
    #     print("pred:", pred.shape, "batch:", bx.shape, by.shape)
    #     annotated_image = TransformDataset.annotate_tensor(bx, pred)
    #     print("output_path:", cfg.paths.output_dir + "/eval_result.png")
    #     torchvision.utils.save_image(annotated_image, cfg.paths.output_dir + "/eval_result.png")
    #     break
    
    #1 image

    annotated_image = eval_image(model=model, image_path='data/ibug_tiny/lfpw/trainset/image_0447.png')
    torchvision.utils.save_image(annotated_image, "eval_result.png")
    return metric_dict, object_dict

    #FOR VIDEO
    import cv2
    import torch
    import albumentations as A
    from albumentations import Compose
    from albumentations.pytorch.transforms import ToTensorV2
    import numpy as np
    # Open the video file
    cap = cv2.VideoCapture('src/video.mp4')

    # Loop over the frames and convert them to tensors
    frames = []
    transform = Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    time = 1
    frame_to_cap = time * 30
    count = 0
    while count < frame_to_cap:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to RGB format and normalize it to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype('float32') / 255.0
        
        # Convert the frame to a PyTorch tensor and add it to the list of frames
        tensor = torch.from_numpy(frame)
        #print (tensor.shape)
        #convert tensor to numpy
        array = tensor.numpy()
        transformed = transform(image=array)
        
        #print(tensor.shape) #batch,3,480,480
        frames.append(tensor)
        count += 1

    # Release the video file handle
    cap.release()

    # Stack the frames into a single tensor with shape (num_frames, channels, height, width)
    video_tensor = torch.stack(frames).permute(0,3,1,2)
    with torch.no_grad():
        output_tensor = model(video_tensor)
    annotated_video = TransformDataset.annotate_tensor(video_tensor, output_tensor)
    torchvision.utils.save_image(annotated_video[0], "eval_result.png")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)

def eval_image(image_path, model):
    
    transform = Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    input_image = Image.open(image_path).convert('RGB')
    input_image = np.array(input_image)
    print(input_image.shape)
    input_tensor = transform(image=input_image)
    input_tensor = input_tensor['image'].unsqueeze(0)
    # print(input_tensor.size())
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # prediction = torch.argmax(output_tensor, dim=1)
    print(input_tensor.shape, output_tensor.shape) #1 3 224 224, 1 68 2
    annotated_image = TransformDataset.annotate_tensor(input_tensor, output_tensor)
    return annotated_image

if __name__ == "__main__":
    main()
