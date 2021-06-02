# UMLs
Embedded Representation of Relational Clinical Codes using Knowledge Graphs and Distributed Vectors

### Create env and install the packages
conda create --name umls python=3.8.4
conda activate umls
pip install -r requirements.txt

### Run preprocessing script to convert the data into proper format

    $python umls_pre_process.py
    
### Run only doc2vec dbow
    $python train.py --data_file_name 'desc_input_train.txt' --rel_data_file_name 'codes_input_train.txt' --rel_full_data_file_name 'codes_input_full.txt' \
    --d2v_model_ver dbow --kg_model_ver none --num_epochs 5 --batch_size 32 --num_noise_words 2 --vec_dim 100 --lr 1e-3 --context_size=0

### Run only transh
    $python train.py --data_file_name 'desc_input_train.txt' --rel_data_file_name 'codes_input_train.txt' --rel_full_data_file_name 'codes_input_full.txt' \
    --d2v_model_ver none --kg_model_ver transh --num_epochs 5 --batch_size 32 --num_noise_words 2 --vec_dim 100 --lr 1e-3 --context_size=0
    
### Run DBOW + transH
    $python train.py --data_file_name 'desc_input_train.txt' --rel_data_file_name 'codes_input_train.txt' --rel_full_data_file_name 'codes_input_full.txt' \
    --d2v_model_ver dbow --kg_model_ver transh --num_epochs 5 --batch_size 32 --num_noise_words 2 --vec_dim 100 --lr 1e-3 --context_size=0
    
### For dm context_size needs to > 0
