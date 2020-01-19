# AMAN

**Aesthetic Multi-Attribute Network (AMAN)** contains Multi-Attribute Feature Network (MAFN), Channel and Spatial
Attention Network (CSAN), and Language Generation Network (LGN). The core ofMAFNcontains GFN and AFN, which regress
the global score and attribute scores of an image in PCCD using multi-task regression. They share the dense feature map and
have separated global and attribute feature maps, respectively. Our AMAN is pre-trained on PCCD and finetuned on our DPCCaptions
dataset. The CSAN dynamically adjusts the attentional weights of channel dimension and spatial dimension [6] of
the extracted features. The LGN generates the final comments by LSTM networks which are fed with ground truth attribute
captions in DPC-Captions and attribute feature maps from CSAN.

<div align="center">
  <img src="https://i.loli.net/2020/01/19/j17EYr8eSnMwkLV.jpg", width='700'><br><br>
</div>

### DPC-Captions Dataset

The aesthetic attributes of image captions are from PCCD[1], which contains comments and a score for each of the 7 aesthetic attributes (including overall impression, etc.). However, the scale of PCCD is quite small (only 4235 images). While the AVA[2] dataset contains 255,530 images with an assessment score distribution for each image. The images and score distributions of AVA dataset are crawled from the website of DPChallenge.com. Their exist comments from multiple reviewers attached for every image. However, the multiple comments are not arranged by aesthetic attributes. We then crawl 330,000 images together with their comments from DPChallenge.com. We call this dataset AVA-Plus.    
Images of DPC-Captions are selected from the AVA-Plus with the help of PCCD datasets. The aesthetic attributes of PCCD dataset include ***Color Lighting***, ***Composition***, ***Depth of Field***, ***Focus***, ***General Impression*** and ***Use of Camera***. For each aesthetic attribute, keywords of top 5 frequency are selected from the captions. We omit the adverbs, prepositions and conjunctions. We combine words with similar meaning such as color and colour, colors and colors. A statistic of the keywords frequency are shown in the table below.


(The datasets can be downloaded in https://github.com/BestiVictory/DPC-Captions/)

### PCCD Dataset
PCCD is a nearly fully annotated dataset, which contains comments and a score for each of the 7 aesthetic attributes (including overall
impression, etc.)

(The datasets in the */data* file.)

### Environment
*The project code needs to run on Maxwell and Pascal architecture graphics cards. Incompatibility issues may occur with newer architecture graphics cards.*

Ubuntu 16.04LTS
Theano==0.6.0
pygpu==0.7.6


### Our Paper  
  
Xin Jin, Le Wu, Geng Zhao, Xiaodong Li, Xiaokun Zhang, Shiming Ge, Dongqing Zou, Bin Zhou, Xinghui Zhou. Aesthetic Attributes Assessment of Images. ACM Multimedia (ACMMM), Nice, France, 21-25 Oct. 2019. **[pdf-HD](http://jinxin.me/downloads/papers/031-MM2019/MM2019-HighRes.pdf)**(31.1MB)  **[pdf-LR](http://jinxin.me/downloads/papers/031-MM2019/MM2019-LowRes.pdf)**(1.11MB) **[arXiv](https://arxiv.org/abs/1907.04983)**(1907.04983)


### Citation

Please cite the ACM Multimedia paper if you use DPC-Captions in your work:

```
@inproceedings{DBLP:conf/mm/JinWZLZGZZZ19,
  author    = {Xin Jin, Le Wu, Geng Zhao, Xiaodong Li, Xiaokun Zhang, Shiming Ge, Dongqing Zou, Bin Zhou and Xinghui Zhou},
  title     = {Aesthetic Attributes Assessment of Images},
  booktitle = {Proceedings of the 27th {ACM} International Conference on Multimedia,
               {MM} 2019, Nice, France, October 21-25, 2019},
  pages     = {311--319},
  year      = {2019},
  crossref  = {DBLP:conf/mm/2019},
  url       = {https://doi.org/10.1145/3343031.3350970},
  doi       = {10.1145/3343031.3350970},
  timestamp = {Fri, 06 Dec 2019 16:44:03 +0100},
  biburl    = {https://dblp.org/rec/bib/conf/mm/JinWZLZGZZZ19},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
