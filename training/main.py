# *** *** *** ***
# Boiler Codes - Import Dependencies

if __name__ == '__main__': # used for Windows freeze_support() issues
    from torch.optim import lr_scheduler
    import torch.nn as nn
    import torch.optim as optim
    import torch
    import torch.multiprocessing
    import numpy as np
    import random
    import copy
    import os, sys, glob, shutil
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import json
    import argparse
    # from torchsummary import summary

    sys.path.insert(0, os.path.abspath('.'))
    from configs.params import *
    from configs import params
    from configs import datasets_config as config
    import data.data_loader as data_loader
    from network.logits import ArcFace
    import network.fsb_hash_net as net
    import train as train
    from network import load_model
    import eval.roc_eval_verification as verification
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Imported.")

    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--method', default=params.method, type=str,
                        help='method (backbone)')
    parser.add_argument('--remarks', default=params.remarks, type=str,
                        help='additional remarks')
    parser.add_argument('--write_log', default=params.write_log, type=bool,
                        help='flag to write logs')
    parser.add_argument('--dim', default=params.dim, type=int, metavar='N',
                        help='embedding dimension')
    parser.add_argument('--hash_dim', default=params.hash_dim, type=int, metavar='N',
                        help='hash dimension')
    parser.add_argument('--epochs', default=params.epochs, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=params.lr, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--w_decay', '--w_decay', default=params.w_decay, type=float,
                        metavar='Weight Decay', help='weight decay')
    parser.add_argument('--dropout', '--dropout', default=params.dropout, type=float,
                        metavar='Dropout', help='dropout probability')
    parser.add_argument('--pretrained', default='/mnt/c/Users/msp/Documents/git-repo/fsb_hashnet/models/pretrained/MobileFaceNet_1024.pt', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none)')

    args = parser.parse_args()

    # Determine if an nvidia GPU is available
    device = params.device

    # For reproducibility, Seed the RNG for all devices (both CPU and CUDA):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    file_main_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    print('Running on device: {}'.format(device))
    start_ = datetime.now()
    start_string = start_.strftime("%Y%m%d_%H%M%S")

    # *** *** *** ***
    # Load Dataset and Display Other Parameters

    # Face images
    face_train_dir = config.trainingdb['face_train']
    face_loader_train, face_train_set = data_loader.gen_data(face_train_dir, 'train_rand', type='face', aug='True')
    face_loader_train_tl, face_train_tl_set = data_loader.gen_data(face_train_dir, 'train', type='face', aug='True')

    # Periocular Images
    peri_train_dir = config.trainingdb['peri_train']
    peri_loader_train, peri_train_set = data_loader.gen_data(peri_train_dir, 'train_rand', type='periocular', aug='True')
    peri_loader_train_tl, peri_train_tl_set = data_loader.gen_data(peri_train_dir, 'train', type='periocular', aug='True')

    # Validation Periocular (Gallery/Val + Probe/Test)
    peri_val_dir = config.trainingdb['peri_val']
    peri_loader_val, peri_val_set = data_loader.gen_data(peri_val_dir, 'test', type='periocular', aug='False')

    # Validation Face (Gallery/Val + Probe/Test)
    face_val_dir = config.trainingdb['face_val']
    face_loader_val, face_val_set = data_loader.gen_data(face_val_dir, 'test', type='face', aug='False')

    # Set and Display all relevant parameters
    print('\n***** Face ( Train ) *****\n')
    face_num_train = len(face_train_set)
    face_num_sub = len(face_train_set.classes)
    print(face_train_set)
    print('Num. of Sub.\t\t:', face_num_sub)
    print('Num. of Train. Imgs (Face) \t:', face_num_train)

    print('\n***** Periocular ( Train ) *****\n')
    peri_num_train = len(peri_train_set)
    peri_num_sub = len(peri_train_set.classes)
    print(peri_train_set)
    print('Num. of Sub.\t\t:', peri_num_sub)
    print('Num. of Train Imgs (Periocular) \t:', peri_num_train)

    print('\n***** Periocular ( Validation ) *****\n')
    peri_num_val = len(peri_val_set)
    print(peri_val_set)
    print('Num. of Sub.\t\t:', len(peri_val_set.classes))
    print('Num. of Validation Imgs (Periocular) \t:', peri_num_val)

    print('\n***** Face ( Validation ) *****\n')
    peri_num_val = len(face_val_set)
    print(face_val_set)
    print('Num. of Sub.\t\t:', len(face_val_set.classes))
    print('Num. of Validation Imgs (Periocular) \t:', peri_num_val)

    print('\n***** Other Parameters *****\n')
    print('Start Time \t\t: ', start_string)
    print('Method (Backbone)\t: ', args.method)
    print('Remarks\t\t\t: ', args.remarks)
    print('Net. Descr.\t\t: ', net_descr)
    print('Seed\t\t\t: ', seed)
    print('Batch Size\t\t: ', batch_size)
    print('Emb. Dimension\t\t: ', args.dim)
    print('Hash. Dimension\t\t: ', args.hash_dim)
    print('# Epoch\t\t\t: ', epochs)
    print('Learning Rate\t\t: ', args.lr)
    print('LR Scheduler\t\t: ', lr_sch)
    print('Weight Decay\t\t: ', args.w_decay)
    print('Dropout Prob.\t\t: ', args.dropout)
    print('BN Flag\t\t\t: ', bn_flag)
    print('BN Momentum\t\t: ', bn_moment)
    print('Scaling\t\t\t: ', af_s)
    print('Margin\t\t\t: ', af_m)
    print('Save Flag\t\t: ', save)
    print('Log Writing\t\t: ', args.write_log)

    # *** *** *** ***
    # Load Pre-trained Model, Define Loss and Other Hyperparameters for Training

    print('\n***** *****\n')
    print('Loading Pretrained Model' )  
    print()

    train_mode = 'eval'
    
    load_model_path_fe = args.pretrained
    feature_extractor = net.FSB_Hash_Net(embedding_size = args.dim, do_prob = args.dropout).eval().to(device)

    state_dict_loaded = feature_extractor.state_dict()
    state_dict_pretrained = torch.load(load_model_path_fe, map_location = device)['state_dict']
    state_dict_temp = {}
    for k in state_dict_loaded:
        if 'encoder' not in k:
            state_dict_temp[k] = state_dict_pretrained['backbone.'+k]
        else:
            print(k, 'not loaded!')
    state_dict_loaded.update(state_dict_temp)
    feature_extractor.load_state_dict(state_dict_loaded)
    del state_dict_loaded, state_dict_pretrained, state_dict_temp

    # 생성자(generator)와 판별자(discriminator) 초기화
    generator = net.Hash_Generator(embedding_size = args.dim, do_prob = args.dropout, device = device, out_embedding_size = hash_dim).eval().to(device)
    discriminator = net.Modality_Discriminator(input_dim=512).eval().to(device)

    # for multiple GPU usage, set device in params to torch.device('cuda') without specifying GPU ID.
    # model = torch.nn.DataParallel(model).cuda()
    ####

    in_features  = feature_extractor.linear.in_features
    out_features = args.hash_dim 

    #### model summary
    # torch.cuda.empty_cache()
    # import os
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # summary(model.to(device),(3,112,112))

    # *** ***

    # print('\n***** *****\n')
    print('Appending Face-FC to model ( w.r.t. Face ) ... ' )  
    feat_fc = ArcFace(in_features = args.dim, out_features = face_num_sub, s = 64.0, m = af_m, device = device).eval().to(device)

    # *** ****

    # print('\n***** *****\n')
    print('Appending Peri-FC to model ( w.r.t. Periocular ) ... ' )
    hash_fc = ArcFace(in_features = args.hash_dim, out_features = face_num_sub, s = 128.0, m = af_m, device = device).eval().to(device)
    
    # **********

    print('Re-Configuring Model, Face-FC, and Peri-FC ... ' ) 
    print()

    # *** ***
    # model : Determine parameters to be freezed, or unfreezed

    for name, param in feature_extractor.named_parameters():
        if epochs_pre > 0:
            param.requires_grad = False # Freezing
            if name in ['linear.weight', 'linear.bias', 
                    'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var'] or 'encoder' in name:
                param.requires_grad = True # Unfreeze these named layers
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True   

    
    for name, layer in feature_extractor.named_modules():
        if isinstance(layer,torch.nn.BatchNorm2d):
            # ***
            layer.momentum = bn_moment
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            if bn_flag == 0 or bn_flag == 1:
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True

    for name, param in generator.named_parameters():
        param.requires_grad = True       

    for name, param in discriminator.named_parameters():
        param.requires_grad = True

    # model : Display all learnable parameters
    for name, param in generator.named_parameters():
        if param.requires_grad:
            print('model (requires grad)\t:', name)            
    # *** 

    # model : Freeze or unfreeze BN parameters
    for name, layer in generator.named_modules():
        if isinstance(layer,torch.nn.BatchNorm2d): # or isinstance(layer,torch.nn.BatchNorm1d):
            # ***
            layer.momentum = bn_moment
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            if bn_flag == 0 or bn_flag == 1:
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True
            # *** 
                
    for name, layer in discriminator.named_modules():
        if isinstance(layer,torch.nn.BatchNorm1d):
            # ***
            layer.momentum = bn_moment
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            if bn_flag == 0 or bn_flag == 1:
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True
            # ***

    # *** ***

    # feat_fc : Determine parameters to be freezed, or unfreezed
    for param in feat_fc.parameters():
        param.requires_grad = True

    # feat_fc : Display all learnable parameters
    print('feat_fc\t:', feat_fc)
    for name, param in feat_fc.named_parameters():
        if param.requires_grad:   
            print('feat_fc\t:', name)

    # *** ***

    # hash_fc : Determine parameters to be freezed, or unfreezed
    for param in hash_fc.parameters():
        param.requires_grad = True
        
    # hash_fc : Display all learnable parameters
    print('hash_fc\t:', hash_fc)
    for name, param in hash_fc.named_parameters():
        if param.requires_grad:
            print('hash_fc\t:', name)

    # ********** 
    # Set an optimizer, scheduler, etc.
    loss_fn = { 'loss_ce' : torch.nn.CrossEntropyLoss(),
                'loss_bce' : nn.BCELoss()}

    parameters_fe = [p for p in feature_extractor.parameters() if p.requires_grad]        
    parameters_generator = [p for p in generator.parameters() if p.requires_grad]
    parameters_discriminator = [p for p in discriminator.parameters() if p.requires_grad]
    parameters_feat_fc = [p for p in feat_fc.parameters() if p.requires_grad]
    parameters_hash_fc = [p for p in hash_fc.parameters() if p.requires_grad]

    optimizer_list = [  {'params': parameters_fe},
                        {'params': parameters_generator},
                        {'params': parameters_feat_fc, 'lr': lr*10, 'weight_decay': args.w_decay},
                        {'params': parameters_hash_fc, 'lr': lr*10, 'weight_decay': args.w_decay},
                    ]

    optimizer_G = optim.AdamW(optimizer_list, lr = args.lr, weight_decay = args.w_decay)
    optimizer_D = optim.AdamW([ {'params': parameters_discriminator, 'lr': lr, 'weight_decay': args.w_decay},
                            ], lr = args.lr, weight_decay = args.w_decay)
    scheduler_G = lr_scheduler.MultiStepLR(optimizer_G, milestones = lr_sch, gamma = 0.1)
    scheduler_D = lr_scheduler.MultiStepLR(optimizer_D, milestones = lr_sch, gamma = 0.1)

    metrics = { 'fps': train.BatchTimer(), 'acc': train.accuracy}

    net_params = { 'time' : start_string, 'network' : net_descr, 'method' : args.method, 'remarks' : args.remarks, 'epochs' : epochs, 
                'face_num_sub' : face_num_sub, 'peri_num_sub': peri_num_sub, 'scale' : af_s, 'margin' : af_m,
                'lr' : args.lr, 'lr_sch': lr_sch, 'w_decay' : args.w_decay, 'dropout' : args.dropout,
                'batch_sub' : batch_sub, 'batch_samp' : batch_samp, 'batch_size' : batch_size, 
                'feat_dims' : args.dim, 'hash_dims' : args.hash_dim, 'seed' : seed }

    # *** *** *** ***
    #### Model Training

    #### Define Logging
    train_mode = 'train'
    log_folder = "./logs/" + str(args.method) + "_" + str(start_string) + "_" + str(args.remarks)
    if not os.path.exists(log_folder) and args.write_log is True:
        os.makedirs(log_folder)
    log_nm = log_folder + "/" + str(args.method) + "_" + str(start_string) + "_" + str(args.remarks) + ".txt"

    # Files Backup
    if args.write_log is True: # only backup if there is log
        # copy main and training files as backup
        for files in glob.glob(os.path.join(file_main_path, '*')):
            if '__' not in files: # ignore __pycache__
                shutil.copy(files, log_folder)
                print(files)
        # networks and logits
        py_extension = '.py'
        desc = file_main_path.split('/')[-1]
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'configs'), 'params' + py_extension), log_folder)
        shutil.copy(os.path.join(file_main_path.replace(file_main_path.split('/')[-1], 'network'), 'fsb_hash_net' + py_extension), log_folder)

    if args.write_log is True:
        file = open(log_nm, 'a+')
        file.write(str(net_descr) + "\n")
        file.write('Training started at ' + str(start_) + ".\n\n")
        file.write('Model parameters: \n')
        file.write(json.dumps(net_params) + "\n\n")
        file.close()

    # *** ***
    #### Training

    best_train_acc = 0
    best_test_acc = 0
    best_epoch = 0
    peri_best_val_acc = 0

    best_generator = copy.deepcopy(generator.state_dict())
    best_feature_extractor = copy.deepcopy(feature_extractor.state_dict())
    best_discriminator = copy.deepcopy(discriminator.state_dict())
    best_feat_fc = copy.deepcopy(feat_fc.state_dict())
    best_hash_fc = copy.deepcopy(hash_fc.state_dict())

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    feature_extractor.eval().to(device)
    generator.eval().to(device)
    discriminator.eval().to(device)
    feat_fc.eval().to(device)
    hash_fc.eval().to(device)

    #### Test before training
    val_peri_eer = verification.val_verify(feature_extractor, generator, config.trainingdb['db_name'], emb_size = args.hash_dim, peri_flag=True, root_drt=config.evaluation['verification'], device=device, mode='stolen')
    val_peri_eer = copy.deepcopy(val_peri_eer)
    print('Val EER - Stolen (Peri)\t: ', val_peri_eer) 

    val_face_eer = verification.val_verify(feature_extractor, generator, config.trainingdb['db_name'], emb_size = args.hash_dim, peri_flag=False, root_drt=config.evaluation['verification'], device=device, mode='stolen')
    val_face_eer = copy.deepcopy(val_face_eer)
    print('Val EER - Stolen (Face)\t: ', val_face_eer) 

    test_peri_eer = verification.im_verify(feature_extractor, generator, emb_size = args.hash_dim, root_drt=config.evaluation['verification'], peri_flag=True, device=device, mode='stolen')
    test_peri_eer = copy.deepcopy(test_peri_eer)
    test_peri_eer = verification.get_avg(test_peri_eer)
    print('Test EER - Stolen (Peri)\t: ', test_peri_eer) 

    test_cross_eer = verification.cm_verify(feature_extractor, generator, emb_size = args.hash_dim, root_drt=config.evaluation['verification'], device=device, mode='stolen')
    test_cross_eer = copy.deepcopy(test_cross_eer)
    test_cross_eer = verification.get_avg(test_cross_eer)
    print('Test EER - Stolen (Cross)\t: ', test_cross_eer)
    

    #### Start Training
    for epoch in range(epochs):    
        print()
        print()        
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        feature_extractor.train().to(device)
        generator.train().to(device)
        discriminator.train().to(device)
        feat_fc.train().to(device)
        hash_fc.train().to(device)        
        
        if bn_flag != 2:
            for layer in feature_extractor.modules():
                if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):                
                    layer.eval()
        
        if epoch + 1 > epochs_pre:
            for name, param in feature_extractor.named_parameters():
                param.requires_grad = True

        if bn_flag != 2:
            for layer in generator.modules():
                if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):                
                    layer.eval()
        
        if bn_flag != 2:
            for layer in discriminator.modules():
                if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):                
                    layer.eval()
                
        train_acc, loss = train.run_train(feature_extractor, generator, discriminator, feat_fc = feat_fc, hash_fc = hash_fc, 
                                            face_loader = face_loader_train, peri_loader = peri_loader_train, face_loader_tl = face_loader_train_tl, peri_loader_tl = peri_loader_train_tl,
                                            net_params = net_params, loss_fn = loss_fn, optimizer_G = optimizer_G, optimizer_D = optimizer_D,
                                            scheduler_G = scheduler_G, scheduler_D = scheduler_D, batch_metrics = metrics, 
                                            show_running = True, device = device, writer = writer)   
        print('Loss : ', loss)

        # *** ***    
        feature_extractor.eval().to(device)
        generator.eval().to(device)
        discriminator.eval().to(device)
        feat_fc.eval().to(device)
        hash_fc.eval().to(device)
        # *****
        
        # # Validation
        val_peri_eer = verification.val_verify(feature_extractor, generator, config.trainingdb['db_name'], emb_size = args.hash_dim, peri_flag=True, root_drt=config.evaluation['verification'], device=device, mode='stolen')
        val_peri_eer = copy.deepcopy(val_peri_eer)
        print('Val EER - Stolen (Peri)\t: ', val_peri_eer) 

        val_face_eer = verification.val_verify(feature_extractor, generator, config.trainingdb['db_name'], emb_size = args.hash_dim, peri_flag=False, root_drt=config.evaluation['verification'], device=device, mode='stolen')
        val_face_eer = copy.deepcopy(val_face_eer)
        print('Val EER - Stolen (Face)\t: ', val_face_eer) 

        test_peri_eer = verification.im_verify(feature_extractor, generator, emb_size = args.hash_dim, root_drt=config.evaluation['verification'], peri_flag=True, device=device, mode='stolen')
        test_peri_eer = copy.deepcopy(test_peri_eer)
        test_peri_eer = verification.get_avg(test_peri_eer)
        print('Test EER - Stolen (Peri)\t: ', test_peri_eer) 

        test_cross_eer = verification.cm_verify(feature_extractor, generator, emb_size = args.hash_dim, root_drt=config.evaluation['verification'], device=device, mode='stolen')
        test_cross_eer = copy.deepcopy(test_cross_eer)
        test_cross_eer = verification.get_avg(test_cross_eer)
        print('Test EER - Stolen (Cross)\t: ', test_cross_eer)
        
        if args.write_log is True:
            file = open(log_nm, 'a+')
            file.write(str('Epoch {}/{}'.format(epoch + 1, epochs)) + "\n")
            file.write('Loss : ' + str(loss) + "\n")
            file.write('Validation EER (Periocular)\t: ' + str(val_peri_eer) + "\n")
            file.write('Test Stolen EER (Peri) \t: ' + str(test_peri_eer) + "\n")
            file.write('Test Stolen EER (Cross) \t: ' + str(test_cross_eer) + "\n\n")
            file.close()  

        if val_peri_eer >= peri_best_val_acc and save == True:
            best_train_acc = train_acc
            peri_best_val_acc = val_peri_eer

            best_feature_extractor = copy.deepcopy(feature_extractor.state_dict())
            best_generator = copy.deepcopy(generator.state_dict())
            best_feat_fc = copy.deepcopy(feat_fc.state_dict())
            best_hash_fc = copy.deepcopy(hash_fc.state_dict())

            print('\n***** *****\n')
            print('Saving Best Model & Rank-1 IR ... ')
            print()
            
            # Set save_best_model_path
            save_best_feature_extractor_dir = './models/best_feature_extractor/'
            if not os.path.exists(save_best_feature_extractor_dir):
                os.makedirs(save_best_feature_extractor_dir)
            
            save_best_generator_dir = './models/best_generator/'
            if not os.path.exists(save_best_generator_dir):
                os.makedirs(save_best_generator_dir)

            save_best_feat_fc_dir = './models/best_feat_fc/'
            if not os.path.exists(save_best_feat_fc_dir):
                os.makedirs(save_best_feat_fc_dir)
                
            save_best_hash_fc_dir = './models/best_hash_fc/'
            if not os.path.exists(save_best_hash_fc_dir):
                os.makedirs(save_best_hash_fc_dir)
                                    
            tag = str(args.method) + '_' + str(args.remarks)

            save_best_feature_extractor_path = save_best_feature_extractor_dir + tag + '_' + str(start_string) + '.pth'
            save_best_generator_path = save_best_generator_dir + tag + '_' + str(start_string) + '.pth'
            save_best_feat_fc_path = save_best_feat_fc_dir + tag + '_' + str(start_string) + '.pth' 
            save_best_hash_fc_path = save_best_hash_fc_dir + tag + '_' + str(start_string) + '.pth' 
                    
            print('Best Model Pth\t: ', save_best_feature_extractor_path)
            print('Best Generator Pth\t: ', save_best_generator_path)
            print('Best Face-FC Pth\t: ', save_best_feat_fc_path)
            print('Best Peri-FC Pth\t: ', save_best_hash_fc_path)

            # *** ***
            
            torch.save(best_feature_extractor, save_best_feature_extractor_path)
            torch.save(best_generator, save_best_generator_path)
            torch.save(best_feat_fc, save_best_feat_fc_path)
            torch.save(best_hash_fc, save_best_hash_fc_path)


    if args.write_log is True:
        file = open(log_nm, 'a+')
        end_ = datetime.now()
        end_string = end_.strftime("%Y%m%d_%H%M%S")
        file.write('Training completed at ' + str(end_) + ".\n\n")
        file.write("Model (Path): " + str(save_best_feature_extractor_path) + "\n\n")    
        file.close()

    # *** *** *** ***
    #### Verification for Test Datasets ( Ethnic, Pubfig, FaceScrub, IMDb Wiki, AR)

    print('\n**** Testing Evaluation (All Datasets) **** \n')

    #### Stolen Token Scenario
    print("EER (Periocular) \n")
    stolen_peri_eer_dict = verification.im_verify(feature_extractor, generator, args.hash_dim, root_drt=config.evaluation['verification'], peri_flag=True, device=device, mode='stolen')
    stolen_peri_eer_dict = copy.deepcopy(stolen_peri_eer_dict)
    print(stolen_peri_eer_dict)

    print("EER (Face) \n")
    stolen_face_eer_dict = verification.im_verify(feature_extractor, generator, args.hash_dim, root_drt=config.evaluation['verification'], peri_flag=False, device=device, mode='stolen')
    stolen_face_eer_dict = copy.deepcopy(stolen_face_eer_dict)
    print(stolen_face_eer_dict)

    print("Cross-Modal EER\n")
    stolen_cm_eer_dict = verification.cm_verify(feature_extractor, generator, emb_size=args.hash_dim, root_drt=config.evaluation['verification'], device=device, mode='stolen')    
    stolen_cm_eer_dict = copy.deepcopy(stolen_cm_eer_dict)
    print(stolen_cm_eer_dict)

    #### User-Specific Token Scenario
    print("EER (Periocular) \n")
    peri_eer_dict = verification.im_verify(feature_extractor, generator, args.hash_dim, root_drt=config.evaluation['verification'], peri_flag=True, device=device, mode='user')
    peri_eer_dict = copy.deepcopy(peri_eer_dict)
    print(peri_eer_dict)

    print("EER (Face) \n")
    face_eer_dict = verification.im_verify(feature_extractor, generator, args.hash_dim, root_drt=config.evaluation['verification'], peri_flag=False, device=device, mode='user')
    face_eer_dict = copy.deepcopy(face_eer_dict)
    print(face_eer_dict)

    print("Cross-Modal EER\n")
    cm_eer_dict = verification.cm_verify(feature_extractor, generator, emb_size=args.hash_dim, root_drt=config.evaluation['verification'], device=device, mode='user')    
    cm_eer_dict = copy.deepcopy(cm_eer_dict)
    print(cm_eer_dict)

    # *** *** *** ***
    # Dataset Performance Summary
    print('**** Testing Summary Results (All Datasets) **** \n')

    # *** ***
    print('\n\n Ethnic \n')

    stolen_ethnic_eer_peri = stolen_peri_eer_dict['ethnic']
    stolen_ethnic_eer_face = stolen_face_eer_dict['ethnic']
    stolen_ethnic_cm_eer = stolen_cm_eer_dict['ethnic']
    ethnic_eer_peri = peri_eer_dict['ethnic']
    ethnic_eer_face = face_eer_dict['ethnic']
    ethnic_cm_eer = cm_eer_dict['ethnic']

    print("Stolen EER (Periocular)\t: ", stolen_ethnic_eer_peri)
    print("Stolen EER (Face)\t: ", stolen_ethnic_eer_face)
    print('Stolen Cross-modal EER \t: ', stolen_ethnic_cm_eer)
    print("EER (Periocular)\t: ", ethnic_eer_peri)
    print("EER (Face)\t: ", ethnic_eer_face)
    print('Cross-modal EER \t: ', ethnic_cm_eer)


    # *** ***
    print('\n\n Pubfig \n')

    stolen_pubfig_eer_peri = stolen_peri_eer_dict['pubfig']
    stolen_pubfig_eer_face = stolen_face_eer_dict['pubfig']
    stolen_pubfig_cm_eer = stolen_cm_eer_dict['pubfig']
    pubfig_eer_peri = peri_eer_dict['pubfig']
    pubfig_eer_face = face_eer_dict['pubfig']
    pubfig_cm_eer = cm_eer_dict['pubfig']

    print("Stolen EER (Periocular)\t: ", stolen_pubfig_eer_peri)
    print("Stolen EER (Face)\t: ", stolen_pubfig_eer_face)
    print('Stolen Cross-modal EER \t: ', stolen_pubfig_cm_eer)
    print("EER (Periocular)\t: ", pubfig_eer_peri)
    print("EER (Face)\t: ", pubfig_eer_face)
    print('Cross-modal EER \t: ', pubfig_cm_eer)


    # *** ***
    print('\n\n FaceScrub\n')

    stolen_facescrub_eer_peri = stolen_peri_eer_dict['facescrub']
    stolen_facescrub_eer_face = stolen_face_eer_dict['facescrub']
    stolen_facescrub_cm_eer = stolen_cm_eer_dict['facescrub']
    facescrub_eer_peri = peri_eer_dict['facescrub']
    facescrub_eer_face = face_eer_dict['facescrub']
    facescrub_cm_eer = cm_eer_dict['facescrub']

    print("Stolen EER (Periocular)\t: ", stolen_facescrub_eer_peri)
    print("Stolen EER (Face)\t: ", stolen_facescrub_eer_face)
    print('Stolen Cross-modal EER \t: ', stolen_facescrub_cm_eer)
    print("EER (Periocular)\t: ", facescrub_eer_peri)
    print("EER (Face)\t: ", facescrub_eer_face)
    print('Cross-modal EER \t: ', facescrub_cm_eer)


    # *** *** *** ***
    print('\n\n IMDB Wiki \n')

    stolen_imdb_wiki_eer_peri = stolen_peri_eer_dict['imdb_wiki']
    stolen_imdb_wiki_eer_face = stolen_face_eer_dict['imdb_wiki']
    stolen_imdb_wiki_cm_eer = stolen_cm_eer_dict['imdb_wiki']
    imdb_wiki_eer_peri = peri_eer_dict['imdb_wiki']
    imdb_wiki_eer_face = face_eer_dict['imdb_wiki']
    imdb_wiki_cm_eer = cm_eer_dict['imdb_wiki']

    print("Stolen EER (Periocular)\t: ", stolen_imdb_wiki_eer_peri)
    print("Stolen EER (Face)\t: ", stolen_imdb_wiki_eer_face)
    print('Stolen Cross-modal EER \t: ', stolen_imdb_wiki_cm_eer)
    print("EER (Periocular)\t: ", imdb_wiki_eer_peri)
    print("EER (Face)\t: ", imdb_wiki_eer_face)
    print('Cross-modal EER \t: ', imdb_wiki_cm_eer)


    # *** *** *** ***
    print('\n\n AR \n')

    stolen_ar_eer_peri = stolen_peri_eer_dict['ar']
    stolen_ar_eer_face = stolen_face_eer_dict['ar']
    stolen_ar_cm_eer = stolen_cm_eer_dict['ar']
    ar_eer_peri = peri_eer_dict['ar']
    ar_eer_face = face_eer_dict['ar']
    ar_cm_eer = cm_eer_dict['ar']

    print("Stolen EER (Periocular)\t: ", stolen_ar_eer_peri)
    print("Stolen EER (Face)\t: ", stolen_ar_eer_face)
    print('Stolen Cross-modal EER \t: ', stolen_ar_cm_eer)
    print("EER (Periocular)\t: ", ar_eer_peri)
    print("EER (Face)\t: ", ar_eer_face)
    print('Cross-modal EER \t: ', ar_cm_eer)

    # *** *** *** ***
    #### Average of all Datasets
    print('\n\n\n Calculating Average \n')

    stolen_avg_peri_eer = verification.get_avg(stolen_peri_eer_dict)
    stolen_avg_face_eer = verification.get_avg(stolen_face_eer_dict)
    stolen_avg_cm_eer = verification.get_avg(stolen_cm_eer_dict)
    avg_peri_eer = verification.get_avg(peri_eer_dict)
    avg_face_eer = verification.get_avg(face_eer_dict)
    avg_cm_eer = verification.get_avg(cm_eer_dict)

    print("Stolen EER (Periocular)\t: ", stolen_avg_peri_eer['avg'], '±', stolen_avg_peri_eer['std'])
    print("Stolen EER (Face)\t: ", stolen_avg_face_eer['avg'], '±', stolen_avg_face_eer['std'])
    print('Stolen Cross-modal EER \t: ', stolen_avg_cm_eer['avg'], '±', stolen_avg_cm_eer['std'])
    print("EER (Periocular)\t: ", avg_peri_eer['avg'], '±', avg_peri_eer['std'])
    print("EER (Face)\t: ", avg_face_eer['avg'], '±', avg_face_eer['std'])
    print('Cross-modal EER \t: ', avg_cm_eer['avg'], '±', avg_cm_eer['std'])


    # *** *** *** ***
    # Write Final Performance Summaries to Log 

    if args.write_log is True:
        file = open(log_nm, 'a+')
        file.write('\n****Stolen:****\n')
        file.write('****Ethnic:****')
        file.write('\nFinal EER. (Periocular): ' + str(stolen_peri_eer_dict['ethnic']) + '\nFinal EER. (Face): ' + str(face_eer_dict['ethnic']))        
        file.write('\nCross-Modal (Ver): ' + str(stolen_cm_eer_dict['ethnic'])+ '\n\n')
        file.write('****Pubfig:****')
        file.write('\nFinal EER. (Periocular): ' + str(stolen_peri_eer_dict['pubfig']) + '\nFinal EER. (Face): ' + str(stolen_face_eer_dict['pubfig']))        
        file.write('\nCross-Modal (Ver): ' + str(stolen_cm_eer_dict['pubfig'])+ '\n\n')
        file.write('****FaceScrub:****')
        file.write('\nFinal EER. (Periocular): ' + str(stolen_peri_eer_dict['facescrub']) + '\nFinal EER. (Face): ' + str(stolen_face_eer_dict['facescrub']))        
        file.write('\nCross-Modal (Ver): ' + str(stolen_cm_eer_dict['facescrub'])+ '\n\n')
        file.write('****IMDB Wiki:****')
        file.write('\nFinal EER. (Periocular): ' + str(stolen_peri_eer_dict['imdb_wiki']) + '\nFinal EER. (Face): ' + str(stolen_face_eer_dict['imdb_wiki']))        
        file.write('\nCross-Modal (Ver): ' + str(stolen_cm_eer_dict['imdb_wiki'])+ '\n\n')
        file.write('****AR:****')
        file.write('\nFinal EER. (Periocular): ' + str(stolen_peri_eer_dict['ar']) + '\nFinal EER. (Face): ' + str(stolen_face_eer_dict['ar']))        
        file.write('\nCross-Modal (Ver): ' + str(stolen_cm_eer_dict['ar'])+ '\n\n') 

        file.write('\n****User:****\n')
        file.write('****Ethnic:****')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['ethnic']) + '\nFinal EER. (Face): ' + str(face_eer_dict['ethnic']))        
        file.write('\nCross-Modal (Ver): ' + str(cm_eer_dict['ethnic'])+ '\n\n')
        file.write('****Pubfig:****')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['pubfig']) + '\nFinal EER. (Face): ' + str(face_eer_dict['pubfig']))        
        file.write('\nCross-Modal (Ver): ' + str(cm_eer_dict['pubfig'])+ '\n\n')
        file.write('****FaceScrub:****')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['facescrub']) + '\nFinal EER. (Face): ' + str(face_eer_dict['facescrub']))        
        file.write('\nCross-Modal (Ver): ' + str(cm_eer_dict['facescrub'])+ '\n\n')
        file.write('****IMDB Wiki:****')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['imdb_wiki']) + '\nFinal EER. (Face): ' + str(face_eer_dict['imdb_wiki']))        
        file.write('\nCross-Modal (Ver): ' + str(cm_eer_dict['imdb_wiki'])+ '\n\n')
        file.write('****AR:****')
        file.write('\nFinal EER. (Periocular): ' + str(peri_eer_dict['ar']) + '\nFinal EER. (Face): ' + str(face_eer_dict['ar']))        
        file.write('\nCross-Modal (Ver): ' + str(cm_eer_dict['ar'])+ '\n\n') 


        file.write('\n\n **** Stolen Average **** \n\n')
        file.write('\nFinal EER. (Periocular): ' + str(stolen_avg_peri_eer['avg']) + ' ± ' + str(stolen_avg_peri_eer['std'])  \
                   + '\nFinal EER. (Face): ' + str(stolen_avg_face_eer['avg']) + ' ± ' + str(stolen_avg_face_eer['std']) )        
        file.write('\nCross-Modal (Ver): ' + str(stolen_avg_cm_eer['avg']) + ' ± ' + str(stolen_avg_cm_eer['std'])  + '\n\n')
        file.write('\n\n **** User Average **** \n\n')
        file.write('\nFinal EER. (Periocular): ' + str(avg_peri_eer['avg']) + ' ± ' + str(avg_peri_eer['std'])  \
                   + '\nFinal EER. (Face): ' + str(avg_face_eer['avg']) + ' ± ' + str(avg_face_eer['std']) )        
        file.write('\nCross-Modal (Ver): ' + str(avg_cm_eer['avg']) + ' ± ' + str(avg_cm_eer['std'])  + '\n\n')
        file.close()

# *** *** *** ***                 