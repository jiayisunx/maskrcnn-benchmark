# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug


def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None, jit=False, int8=False, calibration=False, configure_dir='configure.json', iterations=0, iter_calib=0):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    # generate trace model
    if jit:
        print("generate trace model")
        for i, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids = batch
            with torch.no_grad():
                images = images.to(device)
                traced_backbone = model(images, trace=True)
                break

    # Int8 Calibration
    if int8 and calibration:
        import intel_pytorch_extension as ipex
        print("runing int8 calibration step")
        conf = ipex.AmpConf(torch.int8)
        for i, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids = batch
            with torch.no_grad():
                images = images.to(device)
                with ipex.AutoMixPrecision(conf, running_mode="calibration"):
                    if jit:
                        output = model(images, traced_backbone=traced_backbone)
                    else:
                        output = model(images)
                if iter_calib != 0 and i == iter_calib - 1:
                    break
        conf.save(configure_dir)
    # Inference
    print("runing inference step")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                images = images.to(device)
                if int8:
                    import intel_pytorch_extension as ipex
                    conf = ipex.AmpConf(torch.int8, configure_dir)
                    with ipex.AutoMixPrecision(conf, running_mode="inference"):
                        if jit:
                            output = model(images, traced_backbone=traced_backbone)
                        else:
                            output = model(images)
                else:
                    if jit:
                        output = model(images, traced_backbone=traced_backbone)
                    else:
                        output = model(images)
            if timer:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
        if iterations != 0 and i == iterations - 1:
            break
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        jit=False,
        int8=False,
        calibration=False,
        configure_dir='configure.json',
        iterations=0,
        iter_calib=0
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer, jit, int8, calibration, configure_dir, iterations, iter_calib)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / iter per device, on {} devices)".format(
            total_time_str, total_time * num_devices / iterations, num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / iter per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / iterations,
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
