# Training loop func
from pathlib import Path
import traceback


import torch
import wandb as wb
import torchvision.transforms as T

# My modules
import data
from data.transforms import denormalize_img_transforms
# from data.transforms import make_image_augments
from trainers.schedulers.warmup import GradualWarmupScheduler
from .trainer import Trainer

class TrainerDetr(Trainer):
    def __init__(self,
                 setup, experiment_tracker, dataset=None, data_split={}, 
                 with_norm=True, with_visualization=False):
        super().__init__(setup, experiment_tracker, dataset=dataset, data_split=data_split, 
                         with_norm=with_norm, with_visualization=with_visualization)
        self.denorimalize = denormalize_img_transforms()
    
    def _add_optimizer(self, model_without_ddp):
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": float(self.setup["lr_backbone"]),
            },
        ]

        self.optimizer = torch.optim.AdamW(param_dicts, lr=float(self.setup["lr"]),
                                    weight_decay=float(self.setup["weight_decay"]))
        print('TrainerDetr::Using AdamW optimizer')
    
    def _add_scheduler(self, steps_per_epoch):
        if 'lr_scheduling' in self.setup and self.setup["lr_scheduling"] == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.setup['lr'],
                epochs=self.setup['epochs'],
                steps_per_epoch=steps_per_epoch,
                cycle_momentum=False  # to work with Adam
            )
        elif 'lr_scheduling' in self.setup and self.setup["lr_scheduling"] == "warm_cosine":

            consine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                           T_max=self.setup["epochs"] * steps_per_epoch, 
                                                                           eta_min=0, 
                                                                           last_epoch=-1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=8, total_epoch=5 * steps_per_epoch, after_scheduler=consine_scheduler)

        else:
            self.scheduler = None 
            print('TrainerDetr::Warning::no learning scheduling set')
    
    def use_dataset(self, dataset, split_info):
        """Use specified dataset for training with given split settings"""
        exp_config = self.experiment.in_config
        if 'wrapper' in exp_config["dataset"] and exp_config["dataset"]["wrapper"] is not None:
            datawrapper_class = getattr(data, exp_config["dataset"]["wrapper"])
            print("datawrapper_class", datawrapper_class)
            self.datawraper = datawrapper_class(dataset)
        else:
            self.datawraper = data.RealisticDatasetDetrWrapper(dataset)
        
        # self.datawraper = data.RealisticDatasetDetrWrapper(dataset)
        self.datawraper.load_split(split_info)
        self.datawraper.new_loaders(self.setup['batch_size'], shuffle_train=True, multiprocess=self.setup["multiprocess"])

        if self.standardize_data:
            self.datawraper.standardize_data()

        return self.datawraper
    
    def fit(self, model, model_without_ddp, criterion, rank=0, config=None):
        """Fit provided model to reviosly configured dataset"""

        if not self.datawraper:
            raise RuntimeError('{}::Error::fit before dataset was provided. run use_dataset() first'.format(self.__class__.__name__))
        if self.setup["multiprocess"]:
            self.device = rank
        else:
            self.device = ["cuda:{}".format(did) for did in model.device_ids] if hasattr(model, 'device_ids') \
                                           and len(model.device_ids) > 0 else self.setup['devices']
            self.device = 'cpu' if len(self.device) == 0 else self.device[0]
        
        self._add_optimizer(model_without_ddp)
        self._add_scheduler(len(self.datawraper.loaders.train))
        self.es_tracking = []  # early stopping init
        start_epoch = self._start_experiment(model, config)
        print('{}::NN training Using device: {}'.format(self.__class__.__name__, self.device))
        
        if self.log_with_visualization:
            # to run parent dir -- wandb will automatically keep track of intermediate values
            # Othervise it might only display the last value (if saving with the same name every time)
            self.folder_for_preds = Path('./wandb') / 'intermediate_preds_{}'.format(self.__class__.__name__)
            self.folder_for_preds.mkdir(exist_ok=True)
        
        self._fit_loop_without_matcher(model, criterion, self.datawraper.loaders.train, self.datawraper.loaders.validation, start_epoch=start_epoch)
        print("{}::Finished training".format(self.__class__.__name__))
    

    def _fit_loop_without_matcher(self, model, criterion, train_loader, valid_loader, start_epoch):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""

        global best_valid_loss

        # self.setup["dry_run"] = True
        log_step = wb.run.step - 1
        return_stitches = self.setup["return_stitches"]

        if (self.setup["multiprocess"] and self.device == 0) or not self.setup["multiprocess"]:
            best_valid_loss = self.experiment.last_best_validation_loss()
            best_valid_loss = torch.tensor(best_valid_loss) if best_valid_loss is not None else None
        iter_items = 0
        for epoch in range(start_epoch, wb.config.trainer["epochs"]):
            model.train()
            criterion.train()
            self.datawraper.dataset.set_training(True)
            for i, batch in enumerate(train_loader):
                iter_items += 1
                images, gt = batch['image'], batch['ground_truth']
                images = images.to(self.device)
                outputs = model(images,
                                gt_stitches=gt["masked_stitches"], 
                                gt_edge_mask=gt["stitch_edge_mask"], 
                                return_stitches=return_stitches)
                loss, loss_dict = criterion(outputs, gt, epoch=epoch)

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
                    print(f"epoch: {epoch:02d}, batch: {i:04d}, loss: {loss:.6f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if (self.setup["multiprocess"] and self.device == 0) or not self.setup["multiprocess"]:
                    if self.setup["dry_run"]:
                        break
            
            model.eval()
            criterion.eval()
            self.datawraper.dataset.set_training(False)
            with torch.no_grad():
                valid_losses, valid_loss_dict = [], {}
                for batch in valid_loader:
                    images, gt = batch['image'], batch['ground_truth']
                    images = images.to(self.device)
                    outputs = model(images, 
                                    gt_stitches=gt["masked_stitches"], 
                                    gt_edge_mask=gt["stitch_edge_mask"], 
                                    return_stitches=return_stitches)
                    loss, loss_dict = criterion(outputs, gt, epoch=epoch)
                    valid_losses.append(loss)
                    if len(valid_loss_dict) == 0:
                        valid_loss_dict = {'valid_' + key: [] for key in loss_dict}
                    for key, val in loss_dict.items():
                        if val is not None:
                            valid_loss_dict['valid_' + key].append(val)
                    
                    if (self.setup["multiprocess"] and self.device == 0) or not self.setup["multiprocess"]:
                        if self.setup["dry_run"]:
                            break
                valid_loss = sum(valid_losses) / len(valid_losses)  # Each loss element is already a mean for its batch
                valid_loss_dict = {key: sum(val)/len(val) if len(val) > 0 else None for key, val in valid_loss_dict.items()}

            # Checkpoints: & compare with previous best
            if (self.setup["multiprocess"] and self.device == 0) or not self.setup["multiprocess"]:
                if best_valid_loss is None or valid_loss < best_valid_loss:  # taking advantage of lazy evaluation
                    best_valid_loss = valid_loss
                    self._save_checkpoint(model, epoch, best=True)  # saving only the good models

                else:
                    self._save_checkpoint(model, epoch)

                # Base logging
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!Epoch: {}, Validation Loss: {}'.format(epoch, valid_loss))
                valid_loss_dict.update({'epoch': epoch, 'valid_loss': valid_loss, 'best_valid_loss': best_valid_loss})
                wb.log(valid_loss_dict, step=log_step)

                # prediction for visual reference
                if self.log_with_visualization:
                    self.datawraper.dataset.set_training(False)
                    valid_batch = iter(valid_loader).__next__()
                    self._log_batch_image(model, valid_batch, epoch, log_step, tag="valid", return_stitches=return_stitches)
                 
    def _log_batch_image(self, model, batch_sample, epoch, log_step, tag="valid", return_stitches=False):
        with torch.no_grad():
            try:
                batch_size = 1
                image = batch_sample["image"][:batch_size]
                gt = {key:val[:batch_size] for key, val in batch_sample["ground_truth"].items()}
                name, folder, img_fn = batch_sample["name"][:batch_size], batch_sample["data_folder"][:batch_size], batch_sample["img_fn"][:batch_size]
                inputs = image.to(self.device)

                batch_img_files = self.datawraper.dataset.save_prediction_batch(
                    model(image.to(self.device), gt_stitches=gt["masked_stitches"], gt_edge_mask=gt["stitch_edge_mask"], return_stitches=False), 
                    img_fn, 
                    folder, 
                    save_to=self.folder_for_preds,
                    images=image)
                batch_gt_files = self.datawraper.dataset.save_gt_batch_imgs(
                        gt, img_fn, folder, save_to=self.folder_for_preds
                    )
            except BaseException as e:
                print(e)
                traceback.print_exc()
                print('{}::Error::On saving pattern prediction for image logging. Nothing logged'.format(self.__class__.__name__))
            else:
                print('{}::Logged pattern prediction for {}'.format(self.__class__.__name__, batch_gt_files[0].name))
                try:
                    wb.log({"Input:#{}".format(tag + str(0)): [wb.Image(T.ToPILImage()(batch_sample["image"][0].cpu()))], 'epoch':epoch}, step=log_step)
                    wb.log({"GT: #{}".format(tag + str(0)): [wb.Image(str(batch_gt_files[0]))], 'epoch': epoch}, step=log_step)
                    wb.log({"Output:#{}".format(tag + str(0)): [wb.Image(str(batch_img_files[0]))], 'epoch': epoch}, step=log_step) 
                except BaseException as e:
                        print(e)
                        pass
                

        
