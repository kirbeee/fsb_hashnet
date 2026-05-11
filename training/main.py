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
import eval.roc_eval_verification as verification

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
    pretrained_path: str = '/mnt/c/Users/msp/Documents/git-repo/fsb_hashnet/models/pretrained/MobileFaceNet_1024.pt'
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
        val_peri = verification.val_verify(feature_extractor, generator, config.trainingdb['db_name'],
                                           emb_size=self.cfg.hash_dim, peri_flag=True,
                                           root_drt=config.evaluation['verification'], device=self.device,
                                           mode='stolen')

        val_face = verification.val_verify(feature_extractor, generator, config.trainingdb['db_name'],
                                           emb_size=self.cfg.hash_dim, peri_flag=False,
                                           root_drt=config.evaluation['verification'], device=self.device,
                                           mode='stolen')

        test_peri = verification.im_verify(feature_extractor, generator, emb_size=self.cfg.hash_dim,
                                           root_drt=config.evaluation['verification'], peri_flag=True,
                                           device=self.device, mode='stolen')
        test_peri_avg = verification.get_avg(test_peri)

        test_cross = verification.cm_verify(feature_extractor, generator, emb_size=self.cfg.hash_dim,
                                            root_drt=config.evaluation['verification'], device=self.device,
                                            mode='stolen')
        test_cross_avg = verification.get_avg(test_cross)

        print(f'Val EER (Peri): {val_peri} | Val EER (Face): {val_face}')
        print(f'Test EER (Peri): {test_peri_avg} | Test EER (Cross): {test_cross_avg}')

        return val_peri, test_peri_avg, test_cross_avg

    def evaluate_all_datasets(self, feature_extractor, generator):
        """最終測試階段：針對多個資料集進行完整的評估與結果輸出"""
        print('\n**** Testing Evaluation (All Datasets) **** \n')

        scenarios = ['stolen', 'user']
        results = {}

        # 1. 取得所有 Scenario 的評估結果並印出原始字典
        for mode in scenarios:
            peri_res = verification.im_verify(feature_extractor, generator, self.cfg.hash_dim,
                                              root_drt=config.evaluation['verification'], peri_flag=True,
                                              device=self.device, mode=mode)
            face_res = verification.im_verify(feature_extractor, generator, self.cfg.hash_dim,
                                              root_drt=config.evaluation['verification'],
                                              peri_flag=False, device=self.device, mode=mode)
            cm_res = verification.cm_verify(feature_extractor, generator, emb_size=self.cfg.hash_dim,
                                            root_drt=config.evaluation['verification'],
                                            device=self.device, mode=mode)

            results[f'{mode}_peri'] = peri_res
            results[f'{mode}_face'] = face_res
            results[f'{mode}_cm'] = cm_res

            print("EER (Periocular)\n")
            print(peri_res)
            print("EER (Face)\n")
            print(face_res)
            print("Cross-Modal EER\n")
            print(cm_res)

        # 2. 輸出格式化的摘要 (對應原本的輸出格式)
        print("**** Testing Summary Results (All Datasets) ****\n")

        datasets = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
        dataset_names = ['Ethnic', 'Pubfig', 'FaceScrub', 'IMDB Wiki', 'AR']

        for ds, ds_name in zip(datasets, dataset_names):
            print(f"\n {ds_name}\n")
            print(f"Stolen EER (Periocular) :  {results['stolen_peri'].get(ds)}")
            print(f"Stolen EER (Face)       :  {results['stolen_face'].get(ds)}")
            print(f"Stolen Cross-modal EER  :  {results['stolen_cm'].get(ds)}")
            print(f"EER (Periocular)        :  {results['user_peri'].get(ds)}")
            print(f"EER (Face)      :  {results['user_face'].get(ds)}")
            print(f"Cross-modal EER         :  {results['user_cm'].get(ds)}")

        # 3. 計算並輸出 Average
        print("\n\n Calculating Average\n")

        metrics_to_print = [
            ('stolen_peri', 'Stolen EER (Periocular)'),
            ('stolen_face', 'Stolen EER (Face)'),
            ('stolen_cm', 'Stolen Cross-modal EER'),
            ('user_peri', 'EER (Periocular)'),
            ('user_face', 'EER (Face)'),
            ('user_cm', 'Cross-modal EER')
        ]

        for key, label in metrics_to_print:
            avg_stats = verification.get_avg(results[key])
            print(f"{label} :  {avg_stats.get('avg')} ± {avg_stats.get('std')}")

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
        self.face_loader_train, self.face_train_set = data_loader.gen_data(config.trainingdb['face_train'],
                                                                           'train_rand', type='face', aug='True')
        self.face_loader_train_tl, _ = data_loader.gen_data(config.trainingdb['face_train'], 'train', type='face',
                                                            aug='True')
        self.peri_loader_train, _ = data_loader.gen_data(config.trainingdb['peri_train'], 'train_rand',
                                                         type='periocular', aug='True')
        self.peri_loader_train_tl, _ = data_loader.gen_data(config.trainingdb['peri_train'], 'train', type='periocular',
                                                            aug='True')

        self.face_num_sub = len(self.face_train_set.classes)

    def _build_models(self):
        """初始化並載入權重到所有神經網路組件"""
        print("Loading Models...")
        self.feature_extractor = net.FSB_Hash_Net(embedding_size=self.cfg.dim, do_prob=self.cfg.dropout).to(self.device)

        # 處理特徵提取器的預訓練權重載入
        state_dict_loaded = self.feature_extractor.state_dict()
        state_dict_pretrained = torch.load(self.cfg.pretrained_path, map_location=self.device)['state_dict']
        state_dict_temp = {k: state_dict_pretrained['backbone.' + k] for k in state_dict_loaded if 'encoder' not in k}
        state_dict_loaded.update(state_dict_temp)
        self.feature_extractor.load_state_dict(state_dict_loaded)

        self.generator = net.Hash_Generator(embedding_size=self.cfg.dim, do_prob=self.cfg.dropout, device=self.device,
                                            out_embedding_size=self.cfg.hash_dim).to(self.device)
        self.discriminator = net.Modality_Discriminator(input_dim=512).to(self.device)

        self.feat_fc = ArcFace(in_features=self.cfg.dim, out_features=self.face_num_sub, s=64.0, m=params.af_m,
                               device=self.device).to(self.device)
        self.hash_fc = ArcFace(in_features=self.cfg.hash_dim, out_features=self.face_num_sub, s=128.0, m=params.af_m,
                               device=self.device).to(self.device)

        # 在此實作您原本配置 requires_grad 與 BatchNorm 行為的邏輯
        # self._configure_gradients()

    def _setup_optimizers(self):
        """配置優化器、損失函數與排程器"""
        self.loss_fn = {'loss_ce': nn.CrossEntropyLoss(), 'loss_bce': nn.BCELoss()}

        params_fe = [p for p in self.feature_extractor.parameters() if p.requires_grad]
        params_gen = [p for p in self.generator.parameters() if p.requires_grad]
        params_disc = [p for p in self.discriminator.parameters() if p.requires_grad]
        params_feat_fc = [p for p in self.feat_fc.parameters() if p.requires_grad]
        params_hash_fc = [p for p in self.hash_fc.parameters() if p.requires_grad]

        self.optimizer_G = optim.AdamW([
            {'params': params_fe},
            {'params': params_gen},
            {'params': params_feat_fc, 'lr': self.cfg.lr * 10, 'weight_decay': self.cfg.w_decay},
            {'params': params_hash_fc, 'lr': self.cfg.lr * 10, 'weight_decay': self.cfg.w_decay},
        ], lr=self.cfg.lr, weight_decay=self.cfg.w_decay)

        self.optimizer_D = optim.AdamW([{'params': params_disc, 'lr': params.lr, 'weight_decay': self.cfg.w_decay}],
                                       lr=self.cfg.lr, weight_decay=self.cfg.w_decay)

        self.scheduler_G = lr_scheduler.MultiStepLR(self.optimizer_G, milestones=params.lr_sch, gamma=0.1)
        self.scheduler_D = lr_scheduler.MultiStepLR(self.optimizer_D, milestones=params.lr_sch, gamma=0.1)

    def _set_train_mode(self):
        self.feature_extractor.train()
        self.generator.train()
        self.discriminator.train()
        self.feat_fc.train()
        self.hash_fc.train()

    def _set_eval_mode(self):
        self.feature_extractor.eval()
        self.generator.eval()
        self.discriminator.eval()
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
            # if epoch + 1 > params.epochs_pre: ...

            # 執行一個 epoch 的訓練
            train_acc, loss = train_module.run_train(
                self.feature_extractor, self.generator, self.discriminator,
                feat_fc=self.feat_fc, hash_fc=self.hash_fc,
                face_loader=self.face_loader_train, peri_loader=self.peri_loader_train,
                face_loader_tl=self.face_loader_train_tl, peri_loader_tl=self.peri_loader_train_tl,
                net_params=vars(self.cfg), loss_fn=self.loss_fn,
                optimizer_G=self.optimizer_G, optimizer_D=self.optimizer_D,
                scheduler_G=self.scheduler_G, scheduler_D=self.scheduler_D,
                batch_metrics={'fps': train_module.BatchTimer(), 'acc': train_module.accuracy},
                show_running=True, device=self.device, writer=self.writer
            )

            self._set_eval_mode()
            val_peri_eer, test_peri, test_cross = self.evaluator.validate(self.feature_extractor, self.generator)

            self._log_epoch(epoch, loss, val_peri_eer, test_peri, test_cross)

            if val_peri_eer >= self.best_val_eer and params.save:
                self.best_val_eer = val_peri_eer
                self._save_models()

        # 訓練結束後進行最終的全資料集評估
        self.evaluator.evaluate_all_datasets(self.feature_extractor, self.generator)

    def _log_epoch(self, epoch, loss, val_peri_eer, test_peri, test_cross):
        """將每個 Epoch 的結果寫入日誌"""
        if self.cfg.write_log:
            with open(self.log_file, 'a+') as f:
                f.write(f"Epoch {epoch + 1}/{self.cfg.epochs}\nLoss: {loss}\n")
                f.write(f"Val EER (Peri): {val_peri_eer}\nTest Stolen (Peri): {test_peri}\n\n")

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