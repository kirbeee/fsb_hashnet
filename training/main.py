import os
import sys
import glob
import shutil
import random
import copy
import json
import argparse
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from configs import params
from configs import datasets_config as config
import data.data_loader as data_loader
from network.logits import ArcFace
import network.fsb_hash_net as net
import train as train_module
import eval.plusvein_eval as verification

torch.multiprocessing.set_sharing_strategy('file_system')


# ==========================================
# 1. 配置管理 (Configuration)
# ==========================================
@dataclass
class ExperimentConfig:
    """集中管理所有實驗相關的超參數與設定"""
    method: str = params.method
    remarks: str = params.remarks
    write_log: bool = params.write_log
    dim: int = params.dim
    hash_dim: int = params.hash_dim
    epochs: int = params.epochs
    lr: float = params.lr
    w_decay: float = params.w_decay
    dropout: float = params.dropout
    pretrained_path: str = params.pretrained_path
    device: str = params.device
    seed: int = params.seed

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description='Training Arguments')
        # 這裡可以加入 add_argument 覆蓋 dataclass 的預設值
        parser.add_argument('--method', default=cls.method, type=str)
        parser.add_argument('--remarks', default=cls.remarks, type=str)
        parser.add_argument('--dim', default=cls.dim, type=int)
        parser.add_argument('--hash_dim', default=cls.hash_dim, type=int)
        parser.add_argument('--epochs', default=cls.epochs, type=int)
        parser.add_argument('--lr', default=cls.lr, type=float)
        parser.add_argument('--w_decay', default=cls.w_decay, type=float)
        parser.add_argument('--dropout', default=cls.dropout, type=float)
        parser.add_argument('--pretrained_path', default=cls.pretrained_path, type=str)
        args = parser.parse_args()
        return cls(**vars(args))


# ==========================================
# 2. 實驗環境初始化 (Environment Setup)
# ==========================================
class EnvironmentManager:
    @staticmethod
    def set_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def setup_logging(cfg: ExperimentConfig):
        start_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_folder = f"./logs/{cfg.method}_{start_string}_{cfg.remarks}"
        log_file = os.path.join(log_folder, f"{cfg.method}_{start_string}_{cfg.remarks}.txt")

        if cfg.write_log and not os.path.exists(log_folder):
            os.makedirs(log_folder)
            EnvironmentManager._backup_scripts(log_folder)

        return log_folder, log_file, start_string

    @staticmethod
    def _backup_scripts(log_folder):
        file_main_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        for f in glob.glob(os.path.join(file_main_path, '*.py')):
            shutil.copy(f, log_folder)
        # 您原本備份 configs 與 network 的邏輯也可移至此處


# ==========================================
# 3. 評估器 (Evaluator - 負責各種驗證邏輯)
# ==========================================
class Evaluator:
    """處理模型驗證與測試邏輯，消除重複的程式碼"""

    def __init__(self, cfg: ExperimentConfig, log_file: str):
        self.cfg = cfg
        self.device = cfg.device
        self.log_file = log_file

    def validate(self, feature_extractor, generator):
        """執行 Epoch 內的快速驗證"""
        val_eer = verification.session_verify(
            feature_extractor, generator,
            emb_size=self.cfg.hash_dim,
            root_drt=config.evaluation['verification'],
            enroll_sessions=config.trainingdb['train_sessions'],
            probe_sessions=config.trainingdb['test_sessions'],
            device=self.device,
            mode='stolen',
            class_mode=config.trainingdb['class_mode']
        )

        print(f'Val EER (Session 1→2): {val_eer}')

        return val_eer

    def evaluate_all_datasets(self, feature_extractor, generator):
        """最終測試階段：Session-based 驗證"""
        print('\n**** Testing Evaluation (PLUSVein-FV3) **** \n')

        results = {}
        for mode in ['stolen', 'user']:
            eer_value = verification.session_verify(
                feature_extractor, generator,
                emb_size=self.cfg.hash_dim,
                root_drt=config.evaluation['verification'],
                enroll_sessions=config.trainingdb['train_sessions'],
                probe_sessions=config.trainingdb['test_sessions'],
                device=self.device,
                mode=mode,
                class_mode=config.trainingdb['class_mode']
            )
            results[mode] = eer_value
            print(f"EER ({mode}) : {eer_value}")

        return results


# ==========================================
# 4. 訓練器 (Trainer - 核心狀態機)
# ==========================================
class BiometricTrainer:
    """封裝模型、資料載入、優化器與訓練迴圈"""

    def __init__(self, cfg: ExperimentConfig, log_file: str, start_string: str):
        self.cfg = cfg
        self.device = cfg.device
        self.log_file = log_file
        self.start_string = start_string
        self.writer = SummaryWriter()
        self.writer.iteration = 0
        self.writer.interval = 10
        self.evaluator = Evaluator(cfg, log_file)

        self.best_val_eer = 0.0

        self._prepare_data()
        self._build_models()
        self._setup_optimizers()

    def _prepare_data(self):
        """處理所有 DataLoader 的建立"""
        self.train_loader, self.train_set = data_loader.gen_plusvein_data(
            config.trainingdb['root_dir'],
            mode='train',
            sessions=config.trainingdb['train_sessions'],
            class_mode=config.trainingdb['class_mode'],
            aug='True',
            input_size=(112, 112),
            roi_size=None,
            balanced=True
        )

        self.num_classes = self.train_set.num_classes

    def _configure_gradients(self):
        """實作原始的參數凍結邏輯"""
        for name, param in self.feature_extractor.named_parameters():
            if params.epochs_pre > 0:
                param.requires_grad = False
                if name in ['linear.weight', 'linear.bias',
                            'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var'] or 'encoder' in name:
                    param.requires_grad = True
            else:
                param.requires_grad = True

                # BN 層處理
        for name, layer in self.feature_extractor.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.momentum = params.bn_moment
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                if params.bn_flag == 0 or params.bn_flag == 1:
                    layer.weight.requires_grad = True
                    layer.bias.requires_grad = True

    def _build_models(self):
        """初始化並載入權重到所有神經網路組件"""
        print("Loading Models...")
        self.feature_extractor = net.FSB_Hash_Net(embedding_size=self.cfg.dim, do_prob=self.cfg.dropout).to(self.device)

        # 處理特徵提取器的預訓練權重載入
        if self.cfg.pretrained_path and os.path.exists(self.cfg.pretrained_path):
            state_dict_loaded = self.feature_extractor.state_dict()
            state_dict_pretrained = torch.load(self.cfg.pretrained_path, map_location=self.device)['state_dict']
            state_dict_temp = {}
            skipped = []
            for key in state_dict_loaded:
                if 'encoder' in key:
                    continue
                pretrained_key = 'backbone.' + key
                if pretrained_key in state_dict_pretrained:
                    pretrained_weight = state_dict_pretrained[pretrained_key]
                    if pretrained_weight.shape == state_dict_loaded[key].shape:
                        state_dict_temp[key] = pretrained_weight
                    else:
                        skipped.append((key, state_dict_loaded[key].shape, pretrained_weight.shape))
            state_dict_loaded.update(state_dict_temp)
            self.feature_extractor.load_state_dict(state_dict_loaded)
            for key, expected_shape, actual_shape in skipped:
                print(f"Skipping pretrained weight for {key} (expected {expected_shape}, got {actual_shape}).")
        else:
            print("Skipping pretrained backbone (path not provided or missing).")

        self.generator = net.Hash_Generator(embedding_size=self.cfg.dim, do_prob=self.cfg.dropout, device=self.device,
                                            out_embedding_size=self.cfg.hash_dim).to(self.device)

        self.feat_fc = ArcFace(in_features=self.cfg.dim, out_features=self.num_classes, s=64.0, m=params.af_m,
                               device=self.device).to(self.device)
        self.hash_fc = ArcFace(in_features=self.cfg.hash_dim, out_features=self.num_classes, s=128.0, m=params.af_m,
                               device=self.device).to(self.device)

        # 在此實作您原本配置 requires_grad 與 BatchNorm 行為的邏輯
        self._configure_gradients()

    def _setup_optimizers(self):
        """配置優化器、損失函數與排程器"""
        self.loss_fn = {'loss_ce': nn.CrossEntropyLoss()}

        params_fe = [p for p in self.feature_extractor.parameters() if p.requires_grad]
        params_gen = [p for p in self.generator.parameters() if p.requires_grad]
        params_feat_fc = [p for p in self.feat_fc.parameters() if p.requires_grad]
        params_hash_fc = [p for p in self.hash_fc.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW([
            {'params': params_fe},
            {'params': params_gen},
            {'params': params_feat_fc, 'lr': self.cfg.lr * 10, 'weight_decay': self.cfg.w_decay},
            {'params': params_hash_fc, 'lr': self.cfg.lr * 10, 'weight_decay': self.cfg.w_decay},
        ], lr=self.cfg.lr, weight_decay=self.cfg.w_decay)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=params.lr_sch, gamma=0.1)

    def _set_train_mode(self):
        self.feature_extractor.train()
        self.generator.train()
        self.feat_fc.train()
        self.hash_fc.train()

    def _set_eval_mode(self):
        self.feature_extractor.eval()
        self.generator.eval()
        self.feat_fc.eval()
        self.hash_fc.eval()

    def run(self):
        """主訓練迴圈"""
        # 初步測試 (Test before training)
        self._set_eval_mode()
        self.evaluator.validate(self.feature_extractor, self.generator)

        for epoch in range(self.cfg.epochs):
            print(f'\nEpoch {epoch + 1}/{self.cfg.epochs}')
            print('-' * 10)

            self._set_train_mode()

            # 您原始腳本中解凍網路特定層的邏輯可以放在這裡
            if epoch + 1 > params.epochs_pre:
                for name, param in self.feature_extractor.named_parameters():
                    param.requires_grad = True

            # 執行一個 epoch 的訓練
            train_acc, loss = train_module.run_train(
                self.feature_extractor, self.generator,
                feat_fc=self.feat_fc, hash_fc=self.hash_fc,
                data_loader=self.train_loader,
                net_params=vars(self.cfg), loss_fn=self.loss_fn,
                optimizer=self.optimizer, scheduler=self.scheduler,
                batch_metrics={'fps': train_module.BatchTimer(), 'acc': train_module.accuracy},
                show_running=True, device=self.device, writer=self.writer
            )

            self._set_eval_mode()
            val_eer = self.evaluator.validate(self.feature_extractor, self.generator)

            self._log_epoch(epoch, loss, val_eer)

            if val_eer >= self.best_val_eer and params.save:
                self.best_val_eer = val_eer
                self._save_models()

        # 訓練結束後進行最終的全資料集評估
        self.evaluator.evaluate_all_datasets(self.feature_extractor, self.generator)

    def _log_epoch(self, epoch, loss, val_eer):
        """將每個 Epoch 的結果寫入日誌"""
        if self.cfg.write_log:
            with open(self.log_file, 'a+') as f:
                f.write(f"Epoch {epoch + 1}/{self.cfg.epochs}\nLoss: {loss}\n")
                f.write(f"Val EER (Session 1→2): {val_eer}\n\n")

    def _save_models(self):
        """負責將最佳模型寫入磁碟"""
        print("Saving Best Models...")
        tag = f"{self.cfg.method}_{self.cfg.remarks}_{self.start_string}"

        components = {
            'feature_extractor': self.feature_extractor,
            'generator': self.generator,
            'feat_fc': self.feat_fc,
            'hash_fc': self.hash_fc
        }

        for name, model in components.items():
            dir_path = f'./models/best_{name}/'
            os.makedirs(dir_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(dir_path, f'{tag}.pth'))


# ==========================================
# 5. 程式進入點 (Main Entry Point)
# ==========================================
def main():
    # 1. 取得設定檔
    cfg = ExperimentConfig.from_args()

    # 2. 環境與 Seed 設定
    EnvironmentManager.set_seed(cfg.seed)
    log_folder, log_file, start_string = EnvironmentManager.setup_logging(cfg)

    print(f'Running on device: {cfg.device}')

    # 3. 實例化並執行訓練器
    trainer = BiometricTrainer(cfg, log_file, start_string)
    trainer.run()


if __name__ == '__main__':
    main()
