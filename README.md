<h1 align="center">
    Flexible Secure Biometrics Hashing Network (FSB-HashNet)
</h1>
<h2 align="center">
    Flexible Secure Biometrics: A Protected Modality-Invariant Face-Periocular Recognition System  
</h2>
<h3 align="center">
    Published in Transaction of Information Forensics and Security (DOI: 10.1109/TIFS.2025.3559785) </br>
    <a href="https://ieeexplore.ieee.org/document/10962269"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a>
</h3>
<br/>

![Network Architecture](FSB_HashNet.png?raw=true "FSB-HashNet")
<br/></br>

## Pre-requisites:
- <b>Environment: </b>Check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command for the environment requirement. These files are slightly filtered manually, so there may be redundant packages.
- <b>Dataset: </b> Prepare the PLUSVein-FV3 dataset (training and testing). Ensure the dataset root is located in the `data` directory (e.g., `data/PLUSVein-FV3`) and configure `datasets_config.py` to point to this data directory and session split.
- <b>Pre-trained models: </b>(Optional) The pre-trained MobileFaceNet model for fine-tuning or testing can be downloaded from [this link](https://www.dropbox.com/scl/fi/l3k1h3tc12vy7puargfc3/MobileFaceNet_1024.pt?rlkey=m9zock9slmaivhij6sptjyzl6&st=jy9cb6oj&dl=0).

## Training: 
1. Change hyperparameters accordingly in `params.py` file. The set values used are the default, but it is possible to alternatively change them when running the python file.
2. Run `python training/main.py`. The training should start immediately.
3. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

## Testing:
1. Based on the (pre-)trained models in the `models(/pretrained)` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc. 
2. Performance Evaluation:
    * Verification / Receiver Operating Characteristic (ROC) curve: Run `roc_eval_verification.py`. Based on the generated `.pt` files in `data` directory, run `plot_roc_sota.ipynb` to generate ROC graph. For cases of stolen-token scenario, and user-specific token scenario, refer to the main function in `verification.py` for the configuration.
3. Security and Privacy Analysis: Under the `analysis_privacy_security` folder, run the desired criterion for desired output. Graphs will be generated in the `graphs` directory.
    * Revocability / Unlinkability: In the case of revocability and unlinkability, `generate_scores.py` has to be run first. Then, run the desired criterion `evaluateRevocability.py` or `evaluateUnlinkability.py`.
    * Non-Invertibility: `evaluateNonInvertibility.py` generates the genuine-impostor graph, while `noninvert_calculate_sar.py` calculates the Success Attack Rate (SAR).

## Comparison with State-of-the-Art (SOTA) models

| Method | Intra-Modal EER (%) <br> (Periocular) | Intra-Modal EER (%) <br> (Face) | Cross-Modal EER (%) <br> (Periocular-Face) |
| --- | --- | --- | --- |
| <a href="https://github.com/tiongsikng/cb_net" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">CB-Net</a> <a href="https://ieeexplore.ieee.org/abstract/document/10201879"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/h5tz21big39wd0dzc70ou/AOabrddckd5cKUF3R2p3jw0?rlkey=l8fksw4ekat5jzcgn66jft6n3&st=t1rayruv&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 9.62 | 3.21 | 9.80 |
| <a href="https://github.com/MIS-DevWorks/HA-ViT" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">HA-ViT</a> <a href="https://ieeexplore.ieee.org/document/10068230"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/crjb30rnxe95e6cdbolsk/AFT0bjj1-OzFuRTrictlAuQ?rlkey=rmpe6mriebl5l051pcfatog11&st=os5z2084&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 11.39 | 10.29 | 13.14 |
| <a href="https://github.com/tiongsikng/gc2sa_net" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">GC<sup>2</sup>SA-Net</a> <a href="https://ieeexplore.ieee.org/document/10418204"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/z0sxpfbzmgp76erlcjxij/AIthSVT0Ju6VNeZupjtju1Y?rlkey=k8ivz5l1gv464e4dbxvfu40gc&e=1&st=0yt7hmr1&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 6.39 | 3.14 | 6.50 |
| <a href="https://github.com/MIS-DevWorks/FBR" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">MFA-ViT</a> <a href="https://ieeexplore.ieee.org/document/10656777"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/1guujtv39cpktxk6dknve/ADx9ow2FbTTRMLFGtoKU-yM?rlkey=dzcmdbvjxglu509vgsuexq0ao&st=axkdc096&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 11.18 | 5.17 | 9.41 |
| <a href="https://github.com/tiongsikng/ael_net" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">AELNet</a> <a href="https://www.sciencedirect.com/science/article/pii/S1568494625003552"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/pwc3gnu6vggrtbfwk9vw1/AITjo9pNnqVs8HXfOY2tSGY?rlkey=qujqfhtadnvcxp00zr75nj10m&st=famfx1am&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 6.65 | 3.19 | 6.32 |
| <a href="https://github.com/tiongsikng/fsb_hashnet" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">FSB-HashNet (No Hash)</a> <a href="https://ieeexplore.ieee.org/abstract/document/10962269"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fi/4ifobdj33k43xv45w9zjw/fsb_hashnet.zip?rlkey=i8gpmg9n6vezsofkciuk2zboe&st=icyz1nd5&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 7.10 | 3.26 | 6.77 |
| <a href="https://github.com/tiongsikng/fsb_hashnet" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">FSB-HashNet (Stolen-Key)</a> <a href="https://ieeexplore.ieee.org/abstract/document/10962269"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fi/4ifobdj33k43xv45w9zjw/fsb_hashnet.zip?rlkey=i8gpmg9n6vezsofkciuk2zboe&st=icyz1nd5&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 9.18 | 4.75 | 9.77 |
| <a href="https://github.com/tiongsikng/fsb_hashnet" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">FSB-HashNet (User-Specific)</a> <a href="https://ieeexplore.ieee.org/abstract/document/10962269"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fi/4ifobdj33k43xv45w9zjw/fsb_hashnet.zip?rlkey=i8gpmg9n6vezsofkciuk2zboe&st=icyz1nd5&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 4.24 | 2.11 | 5.26 |

### The project directories are as follows:
<pre>
├── analysis_privacy_security: Contains evaluation for security and privacy analysis - Non-Invertibility, Revocability, Unlinkability.
│   ├── evaluateNonInvertibility - Evaluate non-invertibility criteria via genuine-impostor graph.
│   ├── evaluateRevocability - Evaluate revocability criteria, only can be used after executing <code>generate_scores.py</code>.
│   ├── evaluateUnlinkability - Evaluate unlinkability criteria, only can be used after executing <code>generate_scores.py</code>.
│   ├── generate_scores - Generate genuine-impostor and mated-non-mated scores that are used for revocability and unlinkability evaluation. Scores are generated in <code>graphs/[method]</code> directory.
│   └── noninvert_calculate_sar - Evaluate non-invertibility criteria by calculating the Success Attack Rate (SAR).
├── configs: Contains configuration files and hyperparameters to run the codes.
│   ├── config.py - Contains directory path for dataset files. Change 'main' in 'main_path' dictionary to point to dataset, e.g., <code>/home/fsb_hashnet/data</code> (without slash).
│   └── params.py - Hyperparameters and arguments for training.
├── data: Directory for dataset preprocessing, and folder to insert data based on <code>config.py</code> files.
│   ├── <i><b>[INSERT DATASET HERE.]</i></b>
│   ├── <i>The <code>.pt</code> files to plot the ROC graph will be generated in this directory.</i>
│   └── data_loader.py - PyTorch DataLoader based on a given path and argument (augmentations).
├── eval: Evaluation metrics - Verification. Also contains <code>.ipynb</code> files to plot ROC graphs.
│   ├── plot_roc_sota.ipynb - Notebook to plot ROC curves, based on generated <code>.pt</code> files from <code>verification.py</code>. Graph is generated in <code>graphs</code> directory.
│   └── roc_eval_verification.py - Evaluates Verification Equal Error Rate (EER) and generates Receiver Operating Characteristic (ROC) curve, which are saved as <code>.pt</code> files in <code>data</code> directory. Use these <code>.pt</code> files to generate ROC curve.
├── graphs: Directory where graphs and privacy evaluations are generated.
│   ├── <i>Security and Privacy Analysis graphs and files are generated in this directory.</i>
│   └── <i>ROC curve file is generated in this directory.</i>
├── logs: Directory where logs are generated.
│   └── <i>Logs will be generated in this directory. Each log folder will contain backups of training files with network files and hyperparameters used.</i>
├── models: Directory to store pre-trained models. Trained models are also generated in this directory.
│   ├── <i><b>[INSERT PRE-TRAINED MODELS HERE.]</i></b>
│   ├── <i><b>The base MobileFaceNet for fine-tuning the FSB-HashNet can be downloaded in <a href="https://www.dropbox.com/scl/fi/l3k1h3tc12vy7puargfc3/MobileFaceNet_1024.pt?rlkey=m9zock9slmaivhij6sptjyzl6&st=jy9cb6oj&dl=0">this link</a>.</i></b>
│   └── <i>Trained models will be generated in this directory.</i>
├── network: Contains loss functions and network related files.
│   ├── fsb_hash_net.py - Architecture file for FSB-HashNet. Also used in the case of hashed (stolen token / user-specific token).
│   ├── fsb_hash_net_baseline - Architecture file for baseline FSB-HashNet. Also used in the case of non-hashed baseline.
│   ├── load_model.py - Loads pre-trained weights based on a given model.
│   └── logits.py - Contains some loss functions that are used.
└── <i>training:</i> Main files for training FSB-HashNet.
    ├── main.py - Main file to run for training. Settings and hyperparameters are based on the files in <code>configs</code> directory.
    └── train.py - Training file that is called from <code>main.py</code>. Gets batch of dataloader and contains criterion for loss back-propagation.
</pre>

#### Citation for this work:
```
@ARTICLE{fsb_hashnet,
  author={Ng, Tiong-Sik and Kim, Jihyeon and Teoh, Andrew Beng Jin},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Flexible Secure Biometrics: A Protected Modality-Invariant Face-Periocular Recognition System}, 
  year={2025},
  volume={20},
  number={},
  pages={4610-4621},
  keywords={Face recognition;Feature extraction;Codes;Security;Protection;Generators;Transforms;Transformers;Training;Privacy;Face;periocular;cancellable biometrics;flexible deployment;modality-invariant},
  doi={10.1109/TIFS.2025.3559785}}


```
