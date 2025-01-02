import os
from pathlib import Path
import requests
import time
import json
import yaml
import shutil
from dataclasses import dataclass, field
from typing import Optional
import logging

import torch
from torch import nn
import wandb as wb
from omegaconf import OmegaConf

# My
import data
import models
log = logging.getLogger(__name__)
@dataclass
class WandbConfig:
    wandb_dir: Optional[str] = None
    wandb_cache_dir: Optional[str] = None

@dataclass
class MyExperimentConfig:
    is_training: bool = True
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    local_dir: Optional[str] = None
    wandb_info: WandbConfig = field(default_factory=WandbConfig)

# ------- Class for experiment tracking and information storage -------
class ExperimentWrappper(object):
    """Class provides 
        * a convenient way to store & load experiment info with integration to wandb 
        * some functions & params shortcuts to access wandb functionality
        * for implemented functions, transparent workflow for finished and active (initialized) run 
        * for finished experiments, if w&b run info is specified, it has priority over the localy stored information

        Wrapper currently does NOT wrap one-liners routinely called for active runs like wb.log(), wb.watch()  
    """
    def __init__(
        self, 
        main_config,
        local_output_dir=None, 
        wandb_username='', 
        master_process=False,
        no_sync=False):
        """Init experiment tracking with wandb
            With no_sync==True, run won't sync with wandb cloud. 
            Note that resuming won't work for off-cloud runs as it requiers fetching files from the cloud"""

        self.checkpoint_filetag = 'checkpoint'
        self.final_filetag = 'fin_model_state'
        self.wandb_username = wandb_username
        self.local_output_dir = local_output_dir
        self.master_process=master_process
        self.experiment_config: MyExperimentConfig = main_config.experiment
        self.project = self.experiment_config.project_name
        self.run_name = self.experiment_config.run_name
        self.run_id = self.experiment_config.run_id
        self.run_local_path = self.experiment_config.local_dir
        self.last_checkpoint_step = -1
        if self.experiment_config.is_training and self.run_local_path is not None:
            os.makedirs(self.run_local_path, exist_ok=True)

        self.no_sync = no_sync

        self.in_config = main_config

        # cannot use wb.config, wb.run, etc. until run initialized & logging started & local path
        self.initialized = False  
        self.artifact = None
    
    # ----- start&stop ------
    def init_run(self):
        """Start wandb logging. 
            If run_id is known, run is automatically resumed.
            """
        os.environ['WANDB_DIR'] = self.experiment_config.wandb_info.wandb_dir
        os.environ['WANDB_CACHE_DIR'] = self.experiment_config.wandb_info.wandb_cache_dir
        if self.no_sync:
            os.environ['WANDB_MODE'] = 'dryrun'
            log.warn('run is not synced with wandb cloud')
        if self.master_process:
            config = OmegaConf.to_container(self.in_config, resolve=True, throw_on_missing=True)
            wb.init(
                name=self.run_name, project=self.project, config=config, 
                resume='allow', id=self.run_id,    # Resuming functionality
                dir=self.run_local_path,
                anonymous='allow')
            self.run_id = wb.run.id
            self.team_name = wb.run.get_url().split("/")[3]

            if not self.wandb_username:
                self.wandb_username = wb.run.entity
                log.warn(f'Running wandb in Anonymous Mode. Your temporary account is: {self.wandb_username}')

        self.initialized = True
        self.checkpoint_counter = 0
    
    def stop(self):
        """Stop wandb for current run. All logging finishes & files get uploaded"""
        if self.initialized:
            wb.finish()
        self.initialized = False

    # -------- run info ------
    def full_name(self):
        name = self.project if self.project else ''
        name += ('-' + self.run_name) if self.run_name else ''
        if self.run_id:
            name += ('-' + self.run_id)
        else:
            name += self.in_config.pre_trained if self.in_config.pre_trained is not None else ''
        
        return name
    
    def last_epoch(self):
        """Id of the last epoch processed
            NOTE: part of resuming functionality, only for wandb runs
        """
        run = self._run_object()
        return run.summary['epoch'] if 'epoch' in run.summary else -1
    
    def local_wandb_path(self):
        return self.in_config.experiment.wandb_info.wandb_dir
    
    def data_info(self):
        config = self._run_config()
        split_config = config.data_wrapper.data_split
        data_config = config.data_wrapper.dataset.panel_info

        try:
            self.get_file('data_split.json', os.path.join(config.experiment.wandb_info.wandb_dir, "wandb"))
            split_config.load_from_split_file = os.path.join(config.experiment.wandb_info.wandb_dir, "wandb", 'data_split.json')
        except (ValueError, RuntimeError) as e:  # if file not found, training will just proceed with generated split

            log.warn(f'Skipping loading split file from cloud..')
        
        try:
            self.get_file('panel_classes.json', os.path.join(config.experiment.wandb_info.wandb_dir, "wandb"))
            data_config.panel_classification = os.path.join(config.experiment.wandb_info.wandb_dir, "wandb", 'panel_classes.json')
        except (ValueError, RuntimeError) as e:  # if file not found, training will just proceed with generated split
            log.warn(f'Skipping loading panel classes file from cloud..')

        try:
            self.get_file('param_filter.json', os.path.join(config.experiment.wandb_info.wandb_dir, "wandb"))
            data_config.filter_by_params = os.path.join(config.experiment.wandb_info.wandb_dir, "wandb", 'param_filter.json')
        except (ValueError, RuntimeError) as e:  # if file not found, training will just proceed with given setup
            log.warn(f'Skipping loading parameter filter file from cloud..')

        return split_config, config.data_wrapper.batch_size, data_config
    
    def last_best_validation_loss(self):
        """Id of the last epoch processed
            NOTE: only for experiments running\loaded from wandb
        """
        run = self._run_object()
        return run.summary['best_valid_loss'] if 'best_valid_loss' in run.summary else None

    def NN_config(self):
        """Run configuration params of NeuralNetwork model"""
        config = self._run_config()
        return config['NN']
    
    def add_statistic(self, tag, info, log_msg=''):
        """Add info the run summary (e.g. stats on test set)"""
        
        # Log
        if log_msg:
            log.info(f'Saving statistic <{log_msg}>:')
            log.info(json.dumps(info, sort_keys=True, indent=2) if isinstance(info, dict) else info)

        if not self.run_id:
            log.warn(f'Experiment not connected to the cloud. Statistic {tag} not synced')
            return 

        # different methods for on-going & finished runs
        if self.initialized:
            wb.run.summary[tag] = info
        else:
            if isinstance(info, dict):
                # NOTE Related wandb issue: https://github.com/wandb/client/issues/1934
                for key in info:
                    self.add_statistic(tag + '.' + key, info[key])
            else:
                run = self._run_object()
                run.summary[tag] = info
                run.summary.update()
    
    def add_config(self, tag, info):
        """Add new value to run config. Only for ongoing runs!"""
        if self.initialized:
            wb.config[tag] = info
        else:
            raise RuntimeError('ExperimentWrappper:Error:Cannot add config to finished run')

    def add_artifact(self, path, name, type):
        """Create a new wandb artifact and upload all the contents there"""
        if not self.run_id:
            log.warn(f'Experiment not connected to the cloud. Artifact {name} not synced')
            return 

        path = Path(path)

        if not self.initialized:
            # can add artifacts only to existing runs!
            # https://github.com/wandb/client/issues/1491#issuecomment-726852201
            log.info('Reactivating wandb run to upload an artifact {}!'.format(name))
            wb.init(id=self.run_id, project=self.project, resume='allow')

        artifact = wb.Artifact(name, type=type)
        if path.is_file():
            artifact.add_file(str(path))
        else:
            artifact.add_dir(str(path))
        wb.run.log_artifact(artifact)

        if not self.initialized:
            wb.finish()
    
    def is_finished(self):
        if not self.run_id:
            log.warn(f'Requested status of run not connected to wandb')
            return True
        run = self._run_object()
        return run.state == 'finished'
    
    # ----- finished runs -- help with evaluation ----
    def load_dataset(self, data_root, eval_config={}, unseen=False, batch_size=5, load_all=False):
        """Shortcut to load dataset
        
            NOTE: small default batch size for evaluation even on lightweights machines
        """
        split, _, data_config = self.data_info()
        if unseen:
            load_all = True  # load data as a whole without splitting
            data_config.update(data_folders=data_config['unseen_data_folders'])  # use the unseen folders list
        split = split if not load_all else None

        # Extra evaluation configuration
        data_config.update(eval_config)
        
        # Dataset
        data_class = getattr(data, data_config['class'])
        dataset = data_class(data_root, data_config, 
                             gt_caching=data_config['gt_caching'], 
                             feature_caching=data_config['feature_caching'])
        if 'wrapper' in data_config and data_config["wrapper"] is not None:
            datawrapper_class = getattr(data, data_config["wrapper"])
            datawrapper = datawrapper_class(dataset, known_split=split, batch_size=batch_size)
        else:
            datawrapper = data.RealisticDatasetDetrWrapper(dataset, known_split=split, batch_size=batch_size)

        return dataset, datawrapper
    
    def load_detr_dataset(self, data_root, eval_config={}, unseen=False, batch_size=5, load_all=False):
        """Shortcut to load dataset
        
            NOTE: small default batch size for evaluation even on lightweights machines
        """
        # data_config also contains the names of datasets to use
        split, _, data_config = self.data_info()  # note that run is not initialized -- we use info from finished run
        if unseen:
            load_all = True  # load data as a whole without splitting
            data_config.update(data_folders=data_config['unseen_data_folders'])  # use the unseen folders list
        split = split if not load_all else None

        # Extra evaluation configuration
        data_config.update(eval_config)
        
        # Dataset
        data_class = getattr(data, data_config['class'])
        dataset = data_class(data_root, data_config, gt_caching=True, feature_caching=False)

        datawrapper = data.RealisticDatasetDetrWrapper(dataset, known_split=split, batch_size=batch_size)
        return dataset, datawrapper
    
    def load_detr_model(self, data_config, others=False):
        model, criterion = models.build_former(self.in_config)
        device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        model = nn.DataParallel(model, device_ids=[device])
        criterion.to(device)

        state_dict = self.get_best_model(device=device)['model_state_dict']
        model.load_state_dict(state_dict)
        criterion.print_debug()
        return model, criterion, device
    
    def recover_detr_model(self, data_config=None):
        if not data_config:
            data_config = self.data_info()[-1]
        model, criterion = models.build_former(yaml.safe_load(open(self.in_config["experiment"]["local_path"], "r")))
        # model, criterion = detr_models.build_model({"NN": self.NN_config(), "dataset": data_config})
        device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        model = nn.DataParallel(model, device_ids=[device])
        criterion.to(device)

        state_dict = self.get_best_model(device=device)['model_state_dict']
        model.load_state_dict(state_dict)
        criterion.print_debug()
        return model, criterion, device
    
    def prediction(self, save_to, model, datawrapper, criterion=None, nick='test', sections=['test'], art_name='multi-data', use_gt_stitches=False):
        """Perform inference and save predictions for a given model on a given dataset"""
        prediction_path = datawrapper.predict(model, save_to=save_to, sections=sections, orig_folder_names=True, use_gt_stitches=use_gt_stitches)

        if nick:
            self.add_statistic(nick + '_folder', os.path.basename(prediction_path), log='Prediction save path')

        if art_name:
            art_name = art_name if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0]
            self.add_artifact(prediction_path, art_name, 'result')

        return prediction_path
    
    def prediction_single(self, save_to, model, datawrapper, image, data_name="outside_single"):
        panel_order, panel_idx, prediction_img = datawrapper.predict_single(model, image, data_name, save_to)
        return panel_order, panel_idx, prediction_img
    
    def run_single_img(self, image, model, datawrapper):
        return datawrapper.run_single_img(image, model, datawrapper)
    


    # ---- file info -----
    def checkpoint_filename(self, check_id=None):
        """Produce filename for the checkpoint of given epoch"""
        check_id_str = '_{}'.format(check_id) if check_id is not None else ''
        return '{}{}.pth'.format(self.checkpoint_filetag, check_id_str)

    def artifactname(self, tag, with_version=True, version=None, custom_alias=None):
        """Produce name for wandb artifact for current run with fiven tag"""
        basename = self.run_name + '_' + self.run_id + '_' + tag
        if custom_alias is not None:
            return basename + ':' + custom_alias

        # else -- return a name with versioning
        version_tag = ':v' + str(version) if version is not None else ':latest'
        return basename + version_tag if with_version else basename

    def final_filename(self):
        """Produce filename for the final model file (assuming PyTorch)"""
        return self.final_filetag + '.pth'

    def cloud_path(self):
        """Return virtual path to the current run on wandb could
            Implemented as a function to allow dynamic update of components with less bugs =)
        """
        if not self.run_id:
            raise RuntimeError('ExperimentWrappper:Error:Need to know run id to get path in wandb could')
        project_name = self.project
        run_id = self.run_id
        return self.team_name + "/" + project_name + "/" + run_id
    
    def local_output_path(self):
        return Path(self.local_output_dir)

    def local_artifact_path(self):
        """create & maintain path to save files to-be-commited-as-artifacts"""
        path = self.local_output_path() / 'artifacts' / self.run_id
        if not path.exists():
            path.mkdir(parents=True)
        return path
    
    def get_pre_trained(self):
        return self.in_config.pre_trained

    # ----- working with files -------
    def get_checkpoint_file(self, to_path=None, version=None, device=None):
        """Load checkpoint file for given epoch from the cloud"""
        if not self.run_id:
            raise RuntimeError('ExperimentWrappper:Error:Need to know run id to restore specific checkpoint from the could')
        try:
            art_path = self._load_artifact(self.artifactname('checkpoint', version=version), to_path=to_path)
            for file in art_path.iterdir():  # only one file per checkpoint anyway
                return self._load_model_from_file(file, device)

        except (RuntimeError, requests.exceptions.HTTPError, wb.apis.CommError) as e:  # raised when file is corrupted or not found
            log.error('checkpoint from version \'{}\'is corrupted or lost: {}'.format(version if version else 'latest', e))
            raise e
    
    def get_best_model(self, to_path=None, device=None):
        """Load model parameters (model with best performance) file from the cloud or from locally saved model file if it exists

            NOTE: cloud has a priority as it might contain up-to-date information
        """

        if self.in_config.pre_trained is not None and os.path.exists(self.in_config.pre_trained):
            # local model available
            log.info(f'Loading locally saved model')
            return self._load_model_from_file(self.in_config.pre_trained, device)
        else:
            raise RuntimeError('ExperimentWrappper:Error:Need to know path to the locally saved model ')
    
    def save_checkpoint(self, state, aliases=[], step=0):
        """Save given state dict as torch checkpoint to local run dir
            aliases assign labels to checkpoints for easy retrieval

            NOTE: only for active wandb runs
        """

        if not self.initialized:
            # prevent training updated to finished runs
            raise RuntimeError('Experiment::cannot save checkpoint files to non-active wandb runs')

        log.info('Saving model state -- checkpoint artifact')

        # Using artifacts to store important files for this run
        filename = self.checkpoint_filename(step)
        past_filename = self.checkpoint_filename(self.last_checkpoint_step)
        self.last_checkpoint_step = step
        

        if "best" in aliases:
            best_filename = self.checkpoint_filename('best')
            torch.save(state, self.local_artifact_path() / best_filename)
        if os.path.exists(self.local_artifact_path() / past_filename):
            os.remove(self.local_artifact_path() / past_filename)
        torch.save(state, self.local_artifact_path() / filename)
        
        

    def get_file(self, filename, to_path='.'):
        """Download a file from the wandb experiment to given path or to currect directory"""
        if not self.run_id:
            raise RuntimeError('ExperimentWrappper:Error:Need to know run id to restore a file from the could')
        wb.restore(filename, run_path=self.project + '/' + self.run_id, replace=True, root=to_path)

    # ------- utils -------
    def _load_artifact(self, artifact_name, to_path=None):
        """Download a requested artifact withing current project. Return loaded path"""
        log.info('Requesting artifacts: {}'.format(artifact_name))

        api = wb.Api({'project': self.project})
        artifact = api.artifact(name=artifact_name)
        filepath = artifact.download(str(to_path) if to_path else None)
        log.info('Artifact saved to: {}'.format(filepath))
        return Path(filepath)

    def _run_object(self):
        """ Shortcut for getting reference to wandb api run object. 
            To uniformly access both ongoing & finished runs"""
        return wb.Api().run(self.cloud_path())
    
    def _run_config(self):
        """Shortcut for getting run configuration information"""
        if self.run_id:
            run = wb.Api().run(self.cloud_path())
            return OmegaConf.create(run.config)
        else:
            return self.in_config
            
    def _wait_for_upload(self, artifact_name, max_attempts=10):
        """Wait for an upload of the given version of an artifact"""
        # follows the suggestion of https://github.com/wandb/client/issues/1486#issuecomment-726229978
        log.info('Waiting for artifact {} upload'.format(artifact_name))
        attempt = 1
        while attempt <= max_attempts:
            try:
                time.sleep(5)
                path = self._load_artifact(artifact_name)
                log.info('Requested version is successfully syncronized')
                shutil.rmtree(path, ignore_errors=True)
                break
            except (ValueError, wb.CommError):
                attempt += 1
                log.info('Trying again')
        if attempt > max_attempts:
            log.warn('artifact {} is still not syncronized'.format(artifact_name))

    def _load_model_from_file(self, file, device=None):
        if device is not None:
            return torch.load(file, map_location=device)
        else: 
            return torch.load(file)  # to the same device it was saved from

