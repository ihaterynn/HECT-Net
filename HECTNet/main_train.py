import multiprocessing
import torch
import math
from torch.cuda.amp import GradScaler
from torch.distributed.elastic.multiprocessing import errors

from utils import logger
from options.opts import get_training_arguments
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master, distributed_init
from components.misc.averaging_utils import EMA
from models import get_model
from loss_fn import build_loss_fn
from optim import build_optimizer
from optim.scheduler import build_scheduler
from data import create_train_val_loader
from utils.checkpoint_utils import load_checkpoint, load_model_state
from engine import Trainer
from common import (
    DEFAULT_EPOCHS,
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_EPOCHS,
)

import warnings


@errors.record
def main(opts, **kwargs):
    num_gpus = getattr(opts, "dev.num_gpus", 0)  # defaults are for CPU
    dev_id = getattr(opts, "dev.device_id", torch.device("cpu"))
    device = getattr(opts, "dev.device", torch.device("cpu"))
    is_distributed = getattr(opts, "ddp.use_distributed", False)

    is_master_node = is_master(opts)

    # set-up data loaders
    train_loader, val_loader, train_sampler = create_train_val_loader(opts)

    # REMOVED: All debug code since dataset is confirmed working
    
    # compute max iterations based on max epochs
    # Useful in doing polynomial decay
    is_iteration_based = getattr(opts, "scheduler.is_iteration_based", False)
    if is_iteration_based:
        max_iter = getattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
        if max_iter is None or max_iter <= 0:
            logger.log("Setting max. iterations to {}".format(DEFAULT_ITERATIONS))
            setattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
            max_iter = DEFAULT_ITERATIONS
        setattr(opts, "scheduler.max_epochs", DEFAULT_MAX_EPOCHS)
        if is_master_node:
            logger.log("Max. iteration for training: {}".format(max_iter))
    else:
        max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        if max_epochs is None or max_epochs <= 0:
            logger.log("Setting max. epochs to {}".format(DEFAULT_EPOCHS))
            setattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        setattr(opts, "scheduler.max_iterations", DEFAULT_MAX_ITERATIONS)
        max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        if is_master_node:
            logger.log("Max. epochs for training: {}".format(max_epochs))
    # set-up the model
    model = get_model(opts)

    # memory format
    memory_format = (
        torch.channels_last
        if getattr(opts, "common.channels_last", False)
        else torch.contiguous_format
    )

    if num_gpus == 0:
        logger.warning(
            "No GPUs are available, so training on CPU. Consider training on GPU for faster training"
        )
        model = model.to(device=device, memory_format=memory_format)
    elif num_gpus == 1:
        model = model.to(device=device, memory_format=memory_format)
    elif is_distributed:
        model = model.to(device=device, memory_format=memory_format)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dev_id],
            output_device=dev_id,
            find_unused_parameters=getattr(opts, "ddp.find_unused_params", False),
        )
        if is_master_node:
            logger.log("Using DistributedDataParallel for training")
    else:
        model = model.to(memory_format=memory_format)
        model = torch.nn.DataParallel(model)
        model = model.to(device=device)
        if is_master_node:
            logger.log("Using DataParallel for training")

    # setup criteria
    criteria = build_loss_fn(opts)
    criteria = criteria.to(device=device)

    # create the optimizer
    optimizer = build_optimizer(model, opts=opts)

    # create the gradient scalar
    gradient_scalar = GradScaler(enabled=getattr(opts, "common.mixed_precision", False))

    # LR scheduler
    scheduler = build_scheduler(opts=opts)

    model_ema = None
    use_ema = getattr(opts, "ema.enable", False)

    if use_ema:
        ema_momentum = getattr(opts, "ema.momentum", 0.0001)
        model_ema = EMA(model=model, ema_momentum=ema_momentum, device=device)
        if is_master_node:
            logger.log("Using EMA")

    best_metric = (
        0.0 if getattr(opts, "stats.checkpoint_metric_max", False) else math.inf
    )

    start_epoch = 0
    start_iteration = 0
    resume_loc = getattr(opts, "common.resume", None)
    finetune_loc = getattr(opts, "common.finetune_imagenet1k", None)
    auto_resume = getattr(opts, "common.auto_resume", False)
    if resume_loc is not None or auto_resume:
        (
            model,
            optimizer,
            gradient_scalar,
            start_epoch,
            start_iteration,
            best_metric,
            model_ema,
        ) = load_checkpoint(
            opts=opts,
            model=model,
            optimizer=optimizer,
            model_ema=model_ema,
            gradient_scalar=gradient_scalar,
        )
    elif finetune_loc is not None:
        model, model_ema = load_model_state(opts=opts, model=model, model_ema=model_ema)
        if is_master_node:
            logger.log("Finetuning model from checkpoint {}".format(finetune_loc))

    training_engine = Trainer(
        opts=opts,
        model=model,
        validation_loader=val_loader,
        training_loader=train_loader,
        optimizer=optimizer,
        criterion=criteria,
        scheduler=scheduler,
        start_epoch=start_epoch,
        start_iteration=start_iteration,
        best_metric=best_metric,
        model_ema=model_ema,
        gradient_scalar=gradient_scalar,
    )

    training_engine.run(train_sampler=train_sampler)


def distributed_worker(i, main, opts, kwargs):
    setattr(opts, "dev.device_id", i)
    torch.cuda.set_device(i)
    setattr(opts, "dev.device", torch.device(f"cuda:{i}"))

    ddp_rank = getattr(opts, "ddp.rank", None)
    if ddp_rank is None:  # torch.multiprocessing.spawn
        ddp_rank = kwargs.get("start_rank", 0) + i
        setattr(opts, "ddp.rank", ddp_rank)

    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts, **kwargs)


def main_worker():
    warnings.filterwarnings("ignore")
    
    # Initialize kwargs dictionary
    kwargs = {}
    
    # Get arguments directly without using experiments_config
    opts = get_training_arguments()
    
    # Common settings
    setattr(opts, "common.auto_resume", True)
    setattr(opts, "common.mixed_precision", True)
    setattr(opts, "common.channels_last", False)
    setattr(opts, "common.tensorboard_logging", False)
    setattr(opts, "common.grad_clip", 10.0)
    setattr(opts, "common.accum_freq", 1)
    setattr(opts, "common.results_loc", "hectnet_results")
    
    # Sampler settings (required to fix the resize error)
    setattr(opts, "sampler.name", "batch_sampler")
    setattr(opts, "sampler.bs.crop_size_width", 256)
    setattr(opts, "sampler.bs.crop_size_height", 256)
    
    # Image augmentation settings (required for validation transforms)
    setattr(opts, "image_augmentation.resize.size", 256)
    
    # Add missing image augmentation configurations
    setattr(opts, "image_augmentation.resize.enable", True)
    setattr(opts, "image_augmentation.resize.interpolation", "bicubic")
    setattr(opts, "image_augmentation.center_crop.enable", True)
    setattr(opts, "image_augmentation.center_crop.size", 256)
    setattr(opts, "image_augmentation.random_resized_crop.enable", True)
    setattr(opts, "image_augmentation.random_resized_crop.interpolation", "bicubic")
    setattr(opts, "image_augmentation.random_horizontal_flip.enable", True)
    setattr(opts, "image_augmentation.rand_augment.enable", True)
    setattr(opts, "image_augmentation.random_erase.enable", True)
    setattr(opts, "image_augmentation.random_erase.p", 0.25)
    setattr(opts, "image_augmentation.mixup.enable", False)
    setattr(opts, "image_augmentation.mixup.alpha", 0.2)
    setattr(opts, "image_augmentation.cutmix.enable", False)
    setattr(opts, "image_augmentation.cutmix.alpha", 1.0)
    
    # Dataset settings - using your custom Malaysian food dataset
    setattr(opts, "dataset.root_train", "C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/train")
    setattr(opts, "dataset.root_val", "C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/validation")
    setattr(opts, "dataset.name", "food_another")
    setattr(opts, "dataset.category", "classification")
    setattr(opts, "dataset.train_batch_size0", 8)
    setattr(opts, "dataset.val_batch_size0", 8)
    setattr(opts, "dataset.eval_batch_size0", 8)
    setattr(opts, "dataset.workers", 2)
    setattr(opts, "dataset.prefetch_factor", 2)
    setattr(opts, "dataset.persistent_workers", True)
    setattr(opts, "dataset.pin_memory", True)
    
    # Model settings - UPDATED to use new HECTNet multiscale architecture
    setattr(opts, "model.classification.name", "hectnet_multiscale")
    setattr(opts, "model.classification.n_classes", 5)
    
    # HECTNet multiscale specific parameters
    setattr(opts, "model.classification.hectnet_multiscale.width_multiplier", 0.5)
    setattr(opts, "model.classification.hectnet_multiscale.attn_norm_layer", "layer_norm_2d")
    setattr(opts, "model.classification.hectnet_multiscale.no_fuse_local_global_features", False)
    setattr(opts, "model.classification.hectnet_multiscale.no_ca", False)
    
    # Multiscale branch parameters
    setattr(opts, "model.classification.hectnet_multiscale.aux_dim", 32)
    setattr(opts, "model.classification.hectnet_multiscale.gabor_channels", 16)
    setattr(opts, "model.classification.hectnet_multiscale.fused_dim", 160)
    
    # REMOVE these lines (lines 242-245):
    # setattr(opts, "model.classification.ehfr_net.width_multiplier", 0.5)  
    # setattr(opts, "model.classification.ehfr_net.attn_norm_layer", "layer_norm_2d")
    # setattr(opts, "model.classification.ehfr_net.no_fuse_local_global_features", False)
    # setattr(opts, "model.classification.ehfr_net.no_ca", False)
    setattr(opts, "model.classification.activation.name", "hard_swish")
    setattr(opts, "model.normalization.name", "batch_norm")
    setattr(opts, "model.normalization.momentum", 0.1)
    setattr(opts, "model.activation.name", "hard_swish")
    setattr(opts, "model.layer.global_pool", "mean")
    setattr(opts, "model.layer.conv_init", "kaiming_normal")
    setattr(opts, "model.layer.conv_init_std_dev", 0.02)
    setattr(opts, "model.layer.linear_init", "trunc_normal")
    setattr(opts, "model.layer.linear_init_std_dev", 0.02)
    
    # Loss function - add label smoothing
    setattr(opts, "loss.classification.name", "cross_entropy")
    setattr(opts, "loss.classification.label_smoothing", 0.1)
    setattr(opts, "loss.category", "classification")
    
    # Optimizer settings - adjusted for smaller model and longer training
    setattr(opts, "optimizer.name", "adamw")
    setattr(opts, "optimizer.weight_decay", 0.005)  # Reduced further for smaller model
    setattr(opts, "optimizer.no_decay_bn_filter_bias", True)
    setattr(opts, "optimizer.adamw.beta1", 0.9)
    setattr(opts, "optimizer.adamw.beta2", 0.999)
    
    # Training settings - INCREASED epochs and adjusted learning rates
    setattr(opts, "scheduler.name", "cosine")
    setattr(opts, "scheduler.max_epochs", 100)       # Increased from 50 to 100 epochs
    setattr(opts, "scheduler.is_iteration_based", False)
    setattr(opts, "scheduler.warmup_iterations", 1000)  # Increased warmup for longer training
    setattr(opts, "scheduler.warmup_init_lr", 1e-6)
    setattr(opts, "scheduler.cosine.max_lr", 0.0005)  # Reduced learning rate for smaller model
    setattr(opts, "scheduler.cosine.min_lr", 0.00005) # Reduced min learning rate
    
    # Stats settings - ENHANCED logging for class-wise metrics
    setattr(opts, "stats.val", ["loss", "top1", "top5"])
    setattr(opts, "stats.train", ["loss", "top1"])  # Added top1 to training stats
    setattr(opts, "stats.checkpoint_metric", "top1")
    setattr(opts, "stats.checkpoint_metric_max", True)
    
    # Common settings - ENHANCED logging
    setattr(opts, "common.log_freq", 50)  # More frequent logging (every 50 iterations)
    setattr(opts, "common.tensorboard_logging", True)  # Enable TensorBoard logging
    # Change line 286 from:
    # setattr(opts, "common.run_label", "ehfr_net_width025_100epochs")
    # To:
    setattr(opts, "common.run_label", "hectnet_multiscale_width050_100epochs")  # Updated run label
    
    # EMA settings
    setattr(opts, "ema.enable", True)
    setattr(opts, "ema.momentum", 0.0005)
    
    # GPU/CUDA settings - Force CUDA usage if available
    if torch.cuda.is_available():
        setattr(opts, "dev.device", torch.device("cuda"))
        setattr(opts, "dev.num_gpus", torch.cuda.device_count())
        setattr(opts, "dev.device_id", 0)  # Use first GPU
        print(f"CUDA is available! Using {torch.cuda.device_count()} GPU(s)")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will use CPU.")
        setattr(opts, "dev.device", torch.device("cpu"))
        setattr(opts, "dev.num_gpus", 0)
    
    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error("--rank should be >=0. Got {}".format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    world_size = getattr(opts, "ddp.world_size", -1)
    use_distributed = not getattr(opts, "ddp.disable", False)
    if num_gpus <= 1:
        use_distributed = False
    setattr(opts, "ddp.use_distributed", use_distributed)

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    norm_name = getattr(opts, "model.normalization.name", "batch_norm")
    ddp_spawn = not getattr(opts, "ddp.no_spawn", False)
    if use_distributed and ddp_spawn and torch.cuda.is_available():
        # get device id
        dev_id = getattr(opts, "ddp.device_id", None)
        setattr(opts, "dev.device_id", dev_id)

        if world_size == -1:
            logger.log(
                "Setting --ddp.world-size the same as the number of available gpus"
            )
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)

        if dataset_workers == -1 or dataset_workers is None:
            setattr(opts, "dataset.workers", n_cpus // num_gpus)

        start_rank = getattr(opts, "ddp.rank", 0)
        setattr(opts, "ddp.rank", None)
        kwargs["start_rank"] = start_rank
        setattr(opts, "ddp.start_rank", start_rank)
        torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts, kwargs),
            nprocs=num_gpus,
        )
    else:
        if dataset_workers == -1:
            setattr(opts, "dataset.workers", n_cpus)

        if norm_name in ["sync_batch_norm", "sbn"]:
            setattr(opts, "model.normalization.name", "batch_norm")

        # adjust the batch size
        train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
        val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
        setattr(opts, "dataset.train_batch_size0", train_bsize)
        setattr(opts, "dataset.val_batch_size0", val_bsize)
        setattr(opts, "dev.device_id", None)
        main(opts=opts, **kwargs)


if __name__ == "__main__":
    #
    main_worker()
