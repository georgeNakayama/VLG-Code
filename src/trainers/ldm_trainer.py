# Training loop func
from pathlib import Path
import traceback


import torch
import wandb as wb
import torchvision.transforms as T

# My modules
import data
from trainers.schedulers.warmup import GradualWarmupScheduler
from models.diffusion.ldm import Diffuser
from .trainer import Trainer
                    
class TrainerLDM(Trainer):
    def __init__(self,
                 setup, experiment_tracker, dataset=None, data_split={}, 
                 with_norm=True, with_visualization=False):
        super().__init__(setup, experiment_tracker, dataset=dataset, data_split=data_split, 
                         with_norm=with_norm, with_visualization=with_visualization)
    
    def _add_optimizer(self, model_without_ddp: Diffuser):

        self.optimizer = torch.optim.AdamW(model_without_ddp.model.parameters(), lr=float(self.setup["lr"]),
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
            self.datawraper = data.RealisticDatasetPCDWrapper(dataset)
        
        self.datawraper.load_split(split_info)
        self.datawraper.new_loaders(self.setup['batch_size'], shuffle_train=True, multiprocess=self.setup["multiprocess"])

        if self.standardize_data:
            self.datawraper.standardize_data()

        return self.datawraper
    
    def _log_an_image(self, model, loader, epoch, log_step):
        """Log image of one example prediction to wandb.
            If the loader does not shuffle batches, logged image is the same on every step"""
        with torch.no_grad():
            # using one-sample-from-each-of-the-base-folders loader
            single_sample_loader = self.datawraper.loaders.valid_single_per_data
            if single_sample_loader is None:
                print('{}::Error::Suitable loader is not available. Nothing logged'.format(self.__class__.__name__))

            try: 
                img_files = []
                for batch in single_sample_loader:

                    batch_img_files = self.datawraper.dataset.save_prediction_batch(
                        model(batch['features'].to(self.device)), 
                        batch['pcd_fn'], batch['data_folder'],
                        save_to=self.folder_for_preds, images=None)

                    img_files += batch_img_files
            except BaseException as e:
                print(e)
                traceback.print_exc()
                print('{}::Error::On saving pattern prediction for image logging. Nothing logged'.format(self.__class__.__name__))
            else:
                for i in range(len(img_files)):
                    print('{}::Logged pattern prediction for {}'.format(self.__class__.__name__, img_files[i].name))
                    try:
                        wb.log({img_files[i].name: [wb.Image(str(img_files[i]))], 'epoch': epoch}, step=log_step)  # will raise errors if given file is not an image
                    except BaseException as e:
                        print(e)
                        pass
    
    def fit(self, model, model_without_ddp, rank=0, config=None):
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
        
        self._fit_loop_without_matcher(model, self.datawraper.loaders.train, self.datawraper.loaders.validation, start_epoch=start_epoch)
        print("{}::Finished training".format(self.__class__.__name__))
    

    def _fit_loop_without_matcher(self, model, train_loader, valid_loader, start_epoch):
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
            self.datawraper.dataset.set_training(True)
            for i, batch in enumerate(train_loader):
                iter_items += 1
                pcds, gt, image = batch['pcd'], batch['ground_truth'], batch['image']
                pcds = pcds.to(self.device)
                image = image.to(self.device)
                outputs = model(pcds, image)
                loss, loss_dict = model.module.compute_loss(outputs, "train")
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
            self.datawraper.dataset.set_training(False)
            with torch.no_grad():
                valid_losses, valid_loss_dict = [], {}
                for batch in valid_loader:
                    pcds, gt, image = batch['pcd'], batch['ground_truth'], batch['image']
                    pcds = pcds.to(self.device)
                    image = image.to(self.device)
                    outputs = model(pcds, image)
                    loss, loss_dict = model.module.compute_loss(outputs, "val")
                    valid_losses.append(loss)
                    if len(valid_loss_dict) == 0:
                        valid_loss_dict = {key: [] for key in loss_dict}
                    for key, val in loss_dict.items():
                        if val is not None:
                            valid_loss_dict[ key].append(val)
                    
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
                pcd = batch_sample["pcd"][:batch_size]
                img = batch_sample["image"][:batch_size]
                gt = {key:val[:batch_size] for key, val in batch_sample["ground_truth"].items()}
                name, folder = batch_sample["name"][:batch_size], batch_sample["data_folder"][:batch_size]
                cond = img.to(self.device)
                pcd = pcd.to(self.device)
                recon_output = model.module.reconstruct_first_stage(pcd, return_stitches=False)
                output = model.module.sample(cond, self.device, sample_times=1, guidance_scale=7.5, return_intermediates=False, return_stitches=False)[0]
                if "posterior" in output.keys():
                    del output["posterior"]
                if "posterior" in recon_output.keys():
                    del recon_output["posterior"]
                batch_img_files = self.datawraper.dataset.save_prediction_batch(
                    output, 
                    name, 
                    folder, 
                    save_to=self.folder_for_preds,
                    image=img)
                batch_recon_files = self.datawraper.dataset.save_recon_batch(
                    recon_output, 
                    name, 
                    folder, 
                    save_to=self.folder_for_preds,
                    pcds=pcd)
                batch_gt_files = self.datawraper.dataset.save_gt_batch_imgs(
                        gt, name, folder, save_to=self.folder_for_preds
                    )
            except BaseException as e:
                print(e)
                traceback.print_exc()
                print('{}::Error::On saving pattern prediction for image logging. Nothing logged'.format(self.__class__.__name__))
            else:
                print('{}::Logged pattern prediction for {}'.format(self.__class__.__name__, batch_gt_files[0].name))
                try:
                    wb.log({"Input:#{}".format(tag + str(0)): [wb.Image(T.ToPILImage()(img[0].cpu()))], 'epoch':epoch}, step=log_step)
                    wb.log({"GT: #{}".format(tag + str(0)): [wb.Image(str(batch_gt_files[0]))], 'epoch': epoch}, step=log_step)
                    wb.log({"Output:#{}".format(tag + str(0)): [wb.Image(str(batch_img_files[0]))], 'epoch': epoch}, step=log_step) 
                    wb.log({"Reconstruction:#{}".format(tag + str(0)): [wb.Image(str(batch_recon_files[0]))], 'epoch': epoch}, step=log_step) 
                except BaseException as e:
                        print(e)
                        pass
