# contains directories of database

main_path = { 'main' : '/mnt/c/Users/msp/Documents/git-repo/fsb_hashnet/data' }

evaluation = { 'verification' : main_path['main'] }

trainingdb = { 'db_name' : 'trainingdb', 
              'face_train' : main_path['main'] + '/trainingdb/train/face', 'peri_train' : main_path['main'] + '/trainingdb/train/peri', 
              'face_val' : main_path['main'] + '/trainingdb/val/face', 'peri_val' : main_path['main'] + '/trainingdb/val/peri'}

ethnic = { 'db_name' : 'ethnic',  
'face_gallery' : main_path['main'] + '/ethnic/Recognition/gallery/face',  'peri_gallery' : main_path['main'] + '/ethnic/Recognition/gallery/peri',}

pubfig = { 'db_name' : 'pubfig',
'face_gallery' : main_path['main'] + '/pubfig/gallery/face', 'peri_gallery' : main_path['main'] + '/pubfig/gallery/peri',}

facescrub = { 'db_name' : 'facescrub',
'face_gallery' : main_path['main'] + '/facescrub/gallery/face', 'peri_gallery' : main_path['main'] + '/facescrub/gallery/peri',}

imdb_wiki = { 'db_name' : 'imdb_wiki',
'face_gallery' : main_path['main'] + '/imdb_wiki/gallery/face', 'peri_gallery' : main_path['main'] + '/imdb_wiki/gallery/peri',}

ar = { 'db_name' : 'ar',
'face_gallery' : main_path['main'] + '/ar/gallery/face', 'peri_gallery' : main_path['main'] + '/ar/gallery/peri',}