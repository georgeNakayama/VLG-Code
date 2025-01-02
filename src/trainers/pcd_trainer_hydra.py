# Training loop func
from pathlib import Path
from torch import nn
import traceback
from typing import Literal, Optional, List
import logging
log = logging.getLogger(__name__)
import torch
import wandb as wb
from dataclasses import dataclass, field
import hydra
# My modules
import data
from data.data_wrappers.wrapper_pcd_hydra import PCDDatasetWrapper
from .trainer_hydra import Trainer, OptimizerConfig, SchedulerConfig, TrainerConfig
from experiment_hydra import ExperimentWrappper
from models.pcd2garment.garment_pcd_hydra import GarmentPCD


@dataclass
class TrainerPCDConfig(TrainerConfig):
    _target_: str = "trainers.pcd_trainer_hydra.TrainerPCD"
    lr_pcd_encoder: float = 0.
    

class TrainerPCD(Trainer):
    def __init__(
        self,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        experiment_tracker: ExperimentWrappper, 
        data_wrapper: PCDDatasetWrapper, 
        devices: List[int],
        random_seed: int,
        epochs: int,
        dry_run: bool,
        multiprocess: bool,
        lr: float,
        lr_pcd_encoder: float,
        with_visualization: bool,
        return_stitches: bool,
        ):
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            experiment_tracker=experiment_tracker, 
            data_wrapper=data_wrapper,
            devices=devices,
            random_seed=random_seed,
            epochs=epochs,
            dry_run=dry_run,
            multiprocess=multiprocess,
            lr=lr,
            with_visualization=with_visualization,
            return_stitches=return_stitches)
        self.lr_pcd_encoder = lr_pcd_encoder
        
    
    def _add_optimizer(self, model_without_ddp: GarmentPCD):
        optimizable_params = [(n, p) for n, p in model_without_ddp.named_parameters() if "criteria" not in n and p.requires_grad]
        param_dicts = [
            {"params": [p for n, p in optimizable_params if "pcd_encoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in optimizable_params if "pcd_encoder" in n and p.requires_grad],
                "lr": float(self.lr_pcd_encoder),
            },
        ]
        
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(self.optimizer_config, _convert_="partial", params=param_dicts, lr=float(self.lr))
        log.info('Using {} optimizer'.format(self.optimizer_config._target_))
    
    def _add_scheduler(self, steps_per_epoch):
        if self.scheduler_config._target_ is not None:
            self.scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(self.scheduler_config, optimizer=self.optimizer, steps_per_epoch=steps_per_epoch, epochs=self.epochs)
        else:
            self.scheduler = None  # no scheduler
            log.warn('no learning scheduling set')
    
    
    def _log_an_image(self, model, loader, epoch, log_step):
        """Log image of one example prediction to wandb.
            If the loader does not shuffle batches, logged image is the same on every step"""
        with torch.no_grad():
            # using one-sample-from-each-of-the-base-folders loader
            single_sample_loader = self.datawraper.loaders.valid_single_per_data
            if single_sample_loader is None:
                log.error('Suitable loader is not available. Nothing logged'.format(self.__class__.__name__))

            try: 
                img_files = []
                for batch in single_sample_loader:

                    batch_img_files = self.datawraper.dataset.save_prediction_batch(
                        model(batch['features'].to(self.device)), 
                        batch['pcd_fn'], batch['data_folder'],
                        save_to=self.folder_for_preds, images=None)

                    img_files += batch_img_files
            except BaseException as e:
                log.critical(e)
                traceback.print_exc()
                log.error('On saving pattern prediction for image logging. Nothing logged')
            else:
                for i in range(len(img_files)):
                    log.info('Logged pattern prediction for {}'.format(img_files[i].name))
                    try:
                        wb.log({img_files[i].name: [wb.Image(str(img_files[i]))], 'epoch': epoch}, step=log_step)  # will raise errors if given file is not an image
                    except BaseException as e:
                        log.critical(e)
                        pass
    
    def fit(self, model: nn.parallel.DistributedDataParallel, model_without_ddp: GarmentPCD, rank=0):
        """Fit provided model to reviosly configured dataset"""

        if not self.datawraper:
            raise RuntimeError('{}::Error::fit before dataset was provided. run use_dataset() first'.format(self.__class__.__name__))
        if self.multiprocess:
            self.device = rank
        else:
            self.device = ["cuda:{}".format(did) for did in model.device_ids] if hasattr(model, 'device_ids') \
                                           and len(model.device_ids) > 0 else self.setup['devices']
            self.device = 'cpu' if len(self.device) == 0 else self.device[0]
        self._add_optimizer(model_without_ddp)
        self._add_scheduler(len(self.datawraper.loaders.train))
        self.es_tracking = []  # early stopping init
        # TODO
        start_epoch = self._start_experiment(model)
        log.info('NN training Using device: {}'.format(self.device))
        
        if self.log_with_visualization:
            # to run parent dir -- wandb will automatically keep track of intermediate values
            # Othervise it might only display the last value (if saving with the same name every time)
            self.folder_for_preds = self.experiment.local_output_dir
        
        self._fit_loop_without_matcher(model, self.datawraper.loaders.train, self.datawraper.loaders.validation, start_epoch=start_epoch)
        log.info("Finished training")
    

    def _fit_loop_without_matcher(self, model: nn.DataParallel, train_loader, valid_loader, start_epoch):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""

        global best_valid_loss

        # self.setup["dry_run"] = True
        log_step = wb.run.step - 1
        return_stitches = self.return_stitches

        if (self.multiprocess and self.device == 0) or not self.multiprocess:
            best_valid_loss = self.experiment.last_best_validation_loss()
            best_valid_loss = torch.tensor(best_valid_loss) if best_valid_loss is not None else None
        iter_items = 0
        for epoch in range(start_epoch, wb.config.trainer["epochs"]):
            model.train()
            self.datawraper.dataset.set_training(True)
            for i, batch in enumerate(train_loader):
                iter_items += 1
                pcds, gt = batch['pcd'], batch['ground_truth']
                pcds = pcds.to(self.device)
                outputs = model(pcds,
                                gt_stitches=gt["masked_stitches"], 
                                gt_edge_mask=gt["stitch_edge_mask"], 
                                return_stitches=return_stitches, sample_posterior=True)
                loss, loss_dict = model.module.compute_loss(outputs, gt, epoch=epoch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                
                log_step += 1
                loss_dict.update({'epoch': epoch, 
                                  'batch': i, 
                                  'loss': loss, 
                                  'learning_rate': self.optimizer.param_groups[0]['lr']})
                wb.log(loss_dict, step=log_step)
                if iter_items % 10 == 0:
                    log.info(f"epoch: {epoch:02d}, batch: {i:04d}, loss: {loss:.6f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if (self.multiprocess and self.device == 0) or not self.multiprocess:
                    if self.dry_run:
                        break
            
            model.eval()
            self.datawraper.dataset.set_training(False)
            with torch.no_grad():
                valid_losses, valid_loss_dict = [], {}
                for batch in valid_loader:
                    pcds, gt = batch['pcd'], batch['ground_truth']
                    pcds = pcds.to(self.device)
                    outputs = model(pcds, 
                                    gt_stitches=gt["masked_stitches"], 
                                    gt_edge_mask=gt["stitch_edge_mask"], 
                                    return_stitches=return_stitches)
                    loss, loss_dict = model.module.compute_loss(outputs, gt, epoch=epoch)
                    valid_losses.append(loss)
                    if len(valid_loss_dict) == 0:
                        valid_loss_dict = {'valid_' + key: [] for key in loss_dict}
                    for key, val in loss_dict.items():
                        if val is not None:
                            valid_loss_dict['valid_' + key].append(val)
                    
                    if (self.multiprocess and self.device == 0) or not self.multiprocess:
                        if self.dry_run:
                            break
                valid_loss = sum(valid_losses) / len(valid_losses)  # Each loss element is already a mean for its batch
                valid_loss_dict = {key: sum(val)/len(val) if len(val) > 0 else None for key, val in valid_loss_dict.items()}

            # Checkpoints: & compare with previous best
            if (self.multiprocess and self.device == 0) or not self.multiprocess:
                if best_valid_loss is None or valid_loss < best_valid_loss:  # taking advantage of lazy evaluation
                    best_valid_loss = valid_loss
                    self._save_checkpoint(model, epoch, best=True)  # saving only the good models

                else:
                    self._save_checkpoint(model, epoch)

                # Base logging
                log.info('Epoch: {}, Validation Loss: {}'.format(epoch, valid_loss))
                valid_loss_dict.update({'epoch': epoch, 'valid_loss': valid_loss, 'best_valid_loss': best_valid_loss})
                wb.log(valid_loss_dict, step=log_step)

                # prediction for visual reference
                if self.log_with_visualization:
                    self.datawraper.dataset.set_training(False)
                    valid_batch = iter(valid_loader).__next__()
                    self._log_batch_pcd(model, valid_batch, epoch, log_step, tag="valid", return_stitches=return_stitches)
                 
    def _log_batch_pcd(self, model, batch_sample, epoch, log_step, tag="valid", return_stitches=False):
        with torch.no_grad():
            try:
                batch_size = 1
                pcd = batch_sample["pcd"][:batch_size]
                gt = {key:val[:batch_size] for key, val in batch_sample["ground_truth"].items()}
                name, folder, pcd_fn = batch_sample["name"][:batch_size], batch_sample["data_folder"][:batch_size], batch_sample["pcd_fn"][:batch_size]
                inputs = pcd.to(self.device)
                output = model(inputs, gt_stitches=gt["masked_stitches"], gt_edge_mask=gt["stitch_edge_mask"], return_stitches=False)
                if "posterior" in output.keys():
                    del output["posterior"]
                batch_img_files = self.datawraper.dataset.save_recon_batch(
                    output, 
                    pcd_fn, 
                    folder, 
                    save_to=self.folder_for_preds,
                    pcds=pcd)
                batch_gt_files = self.datawraper.dataset.save_gt_batch_imgs(
                        gt, pcd_fn, folder, save_to=self.folder_for_preds
                    )
            except BaseException as e:
                log.critical(e)
                traceback.print_exc()
                log.error('On saving pattern prediction for image logging. Nothing logged')
            else:
                log.info('Logged pattern prediction for {}'.format(batch_gt_files[0].name))
                try:
                    wb.log({"Input:#{}".format(tag + str(0)): [wb.Object3D(batch_sample["pcd"][0].cpu().numpy())], 'epoch':epoch}, step=log_step)
                    wb.log({"GT: #{}".format(tag + str(0)): [wb.Image(str(batch_gt_files[0]))], 'epoch': epoch}, step=log_step)
                    wb.log({"Output:#{}".format(tag + str(0)): [wb.Image(str(batch_img_files[0]))], 'epoch': epoch}, step=log_step) 
                except BaseException as e:
                        log.critical(e)
                        pass