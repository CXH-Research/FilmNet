### [A Large-scale Film Style Dataset for Learning Multi-frequency Driven Film](https://arxiv.org/abs/2301.08880)

<div>
<span class="author-block">
  <a href='https://zinuoli.github.io/'>Zinuo Li</a><sup> 👨‍💻‍ </sup>
</span>,
  <span class="author-block">
    <a href='https://cxh.netlify.app/'> Xuhang Chen</a><sup> 👨‍💻‍ </sup>
  </span>,
  <span class="author-block">
    <a href="https://people.ucas.edu.cn/~wangshuqiang?language=en" target="_blank">Shuqiang Wang</a><sup> 📮</sup>
  </span> and
  <span class="author-block">
  <a href="https://www.cis.um.edu.mo/~cmpun/" target="_blank">Chi-Man Pun</a><sup> 📮</sup>
</span>
  ( 👨‍💻‍ Equal contributions, 📮 Corresponding )
  </div>

<b>University of Macau, SIAT CAS</b>

2023 INTERNATIONAL JOINT CONFERENCE ON ARTIFICIAL INTELLIGENCE (IJCAI 2023)

[Website & Dataset](https://cxh-research.github.io/FilmNet/) | [Code](https://github.com/CXH-Research/FilmNet) | [Filmset (Onedrive)](https://uofmacau-my.sharepoint.com/personal/yc17491_umac_mo/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyc17491%5Fumac%5Fmo%2FDocuments%2Fdataset%2FFilmSet&ga=1) | [Filmset (Baidu Netdisk)](https://pan.baidu.com/s/1KdXxWkWu5iWKXEl7q-ZTUQ?pwd=03ji)
---

![image](https://github.com/CXH-Research/FilmNet/assets/94612909/a5ce8c39-d4a2-4e2a-87c7-e8688cf020c5)


### Installation
```
git clone https://github.com/CXH-Research/FilmNet.git
cd FilmNet
pip install -r requirements.txt
```

### Training
You may first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TRAINING in traning.yml

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```

### Inference
You may first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in traning.yml
```
python test.py
```

### Acknowledgments
If you find our work helpful for your research, please cite:
```bib
@inproceedings{ijcai2023p129,
  title     = {A Large-Scale Film Style Dataset for Learning Multi-frequency Driven Film Enhancement},
  author    = {Li, Zinuo and Chen, Xuhang and Wang, Shuqiang and Pun, Chi-Man},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {1160--1168},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/129},
  url       = {https://doi.org/10.24963/ijcai.2023/129},
}
```


