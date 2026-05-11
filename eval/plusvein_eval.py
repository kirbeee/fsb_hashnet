import os
import sys
import torch
import numpy as np
import torch.utils.data as data
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.abspath('.'))
from data.data_loader import PlusVeinFV3Dataset, build_plusvein_transforms
from configs import datasets_config as config

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

batch_size = 256

def compute_eer(fpr, tpr):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return np.around(eer, 4)

def _collect_embeddings(dloader, feature_extractor, generator, device, mode):
    embeddings = []
    labels = []
    feature_extractor = feature_extractor.eval().to(device)
    generator = generator.eval().to(device)

    with torch.no_grad():
        for images, lbl in dloader:
            images = images.to(device)
            lbl = lbl.to(device)
            if mode == 'stolen':
                lbl = torch.zeros_like(lbl)
            feature = feature_extractor(images)
            hash_code = generator(feature, lbl)
            embeddings.append(hash_code.detach())
            labels.append(lbl.detach())

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings, labels

def session_verify(feature_extractor, generator, emb_size=512,
                   root_drt=config.evaluation['verification'], device='cuda:0',
                   enroll_sessions=None, probe_sessions=None, mode='user',
                   class_mode=None, input_size=(112, 112), roi_size=None):
    if enroll_sessions is None:
        enroll_sessions = config.trainingdb.get('train_sessions', [1])
    if probe_sessions is None:
        probe_sessions = config.trainingdb.get('test_sessions', [2])
    if class_mode is None:
        class_mode = config.trainingdb.get('class_mode', 'subject_finger')

    transform = build_plusvein_transforms(input_size, roi_size=roi_size, augment=False)

    enroll_set = PlusVeinFV3Dataset(root_drt, sessions=enroll_sessions, class_mode=class_mode,
                                    transform=transform, input_size=input_size, roi_size=roi_size)
    probe_set = PlusVeinFV3Dataset(root_drt, sessions=probe_sessions, class_mode=class_mode,
                                   transform=transform, input_size=input_size, roi_size=roi_size)

    if len(enroll_set) == 0 or len(probe_set) == 0:
        raise ValueError('Enrollment or probe set is empty. Check PLUSVein-FV3 paths and session split.')

    enroll_loader = data.DataLoader(enroll_set, batch_size=batch_size, shuffle=False, num_workers=4)
    probe_loader = data.DataLoader(probe_set, batch_size=batch_size, shuffle=False, num_workers=4)

    enroll_emb, enroll_labels = _collect_embeddings(enroll_loader, feature_extractor, generator, device, mode)
    probe_emb, probe_labels = _collect_embeddings(probe_loader, feature_extractor, generator, device, mode)

    score_mat = torch.matmul(probe_emb, enroll_emb.t()).cpu().numpy()
    probe_labels_np = probe_labels.cpu().numpy()
    enroll_labels_np = enroll_labels.cpu().numpy()

    gen_mask = probe_labels_np[:, None] == enroll_labels_np[None, :]
    gen_scores = score_mat[gen_mask]
    imp_scores = score_mat[~gen_mask]

    if gen_scores.size == 0 or imp_scores.size == 0:
        raise ValueError('Insufficient genuine or impostor pairs for EER computation.')

    y_gen = np.ones(gen_scores.shape[0])
    y_imp = np.zeros(imp_scores.shape[0])
    scores = np.concatenate((gen_scores, imp_scores))
    y = np.concatenate((y_gen, y_imp))

    fpr_tmp, tpr_tmp, _ = roc_curve(y, scores)
    return compute_eer(fpr_tmp, tpr_tmp)
