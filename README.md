# MMN-STVG

1 Download the compressed file of pre-extracted tube features and pre-generated tube-level labels from [One Drive](https://mugei-my.sharepoint.com/:u:/g/personal/x2174_mac2019_vip/EaBHF8NjTLJKmM_u0Qq4pxIB3Ji8SsGe8N6e8k78MskbmQ?e=uzcVwI) , decompress it and modify the stvg-related paths in [tan.config.paths_catalog.py](https://github.com/atao99/MMN-STVG/blob/main/tan/config/paths_catalog.py) accoding to the path of the directory. 

2 Train and val: 

`./scripts/stvg_train.sh`

3 Val only: 

`./scripts/stvg_val.sh`

4 Test: 

`./scripts/stvg_test.sh` 

A result json file will be generated.
