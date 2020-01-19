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

**The DPC-Captions**

The aesthetic attributes of image captions are from PCCD[1], which contains comments and a score for each of the 7 aesthetic attributes (including overall impression, etc.). However, the scale of PCCD is quite small (only 4235 images). While the AVA[2] dataset contains 255,530 images with an assessment score distribution for each image. The images and score distributions of AVA dataset are crawled from the website of DPChallenge.com. Their exist comments from multiple reviewers attached for every image. However, the multiple comments are not arranged by aesthetic attributes. We then crawl 330,000 images together with their comments from DPChallenge.com. We call this dataset AVA-Plus.    
Images of DPC-Captions are selected from the AVA-Plus with the help of PCCD datasets. The aesthetic attributes of PCCD dataset include ***Color Lighting***, ***Composition***, ***Depth of Field***, ***Focus***, ***General Impression*** and ***Use of Camera***. For each aesthetic attribute, keywords of top 5 frequency are selected from the captions. We omit the adverbs, prepositions and conjunctions. We combine words with similar meaning such as color and colour, colors and colors. A statistic of the keywords frequency are shown in the table below.


(The datasets can be downloaded in https://github.com/BestiVictory/DPC-Captions/)

