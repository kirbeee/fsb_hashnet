# contains directories of database

main_path = { 'main' : '/mnt/c/Users/msp/Documents/git-repo/fsb_hashnet/data' }

plusvein_fv3 = {
    'db_name': 'plusvein_fv3',
    'root_dir': main_path['main'] + '/PLUSVein-FV3'
}

evaluation = { 'verification' : plusvein_fv3['root_dir'] }

trainingdb = {
    'db_name': plusvein_fv3['db_name'],
    'root_dir': plusvein_fv3['root_dir'],
    'train_sessions': [1],
    'test_sessions': [2],
    'class_mode': 'subject_finger'
}
