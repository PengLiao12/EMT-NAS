# [EMT-NAS:Transferring Architectural Knowledge Between Tasks From Different Datasets (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Liao_EMT-NASTransferring_Architectural_Knowledge_Between_Tasks_From_Different_Datasets_CVPR_2023_paper.html)
```
@InProceedings{Liao_2023_CVPR,
    author    = {Liao, Peng and Jin, Yaochu and Du, Wenli},
    title     = {EMT-NAS:Transferring Architectural Knowledge Between Tasks From Different Datasets},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {3643-3653}
}
```
If you find this code helpful for your research, please cite our paper.

# File structure of this example
+ Path of the datasets:

  - `./dataset/`

+ Search space and algorithm related：

  - `./gnas/`

  - `./models/`

  - `./modules/`

  - `cnn_utils.py`

  - `common.py`

  - `data.py`

+ Parameter settings：

  - `config.py`

+ Resulting output folder:

  - `./logs/`

+ Master file：

  -  `main.py`
 
# Requirements
| Package   | Version  |  Note|
| :------------- | :----------: | :----------: | 
| python |   3.6.8   | 
| torch |   1.10.2+cu113   | 
| torchaudio  |    0.10.2+cu113     |    
| torchvision  |    0.11.3+cu113     |   
| graphviz   | 0.14.2|Drawing the structure of the cells |
| pygraphviz | 1.6 |Drawing the structure of the cells |
# Run

The two tasks running by default are CIFAR-10 and CIFAR-100:

```
python main.py 
```
