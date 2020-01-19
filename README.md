# AMAN

*Aesthetic Multi-Attribute Network (AMAN)* contains Multi-Attribute Feature Network (MAFN), Channel and Spatial
Attention Network (CSAN), and Language Generation Network (LGN). The core ofMAFNcontains GFN and AFN, which regress
the global score and attribute scores of an image in PCCD using multi-task regression. They share the dense feature map and
have separated global and attribute feature maps, respectively. Our AMAN is pre-trained on PCCD and finetuned on our DPCCaptions
dataset. The CSAN dynamically adjusts the attentional weights of channel dimension and spatial dimension [6] of
the extracted features. The LGN generates the final comments by LSTM networks which are fed with ground truth attribute
captions in DPC-Captions and attribute feature maps from CSAN.

<div align="center">
  <img src="https://i.loli.net/2020/01/19/j17EYr8eSnMwkLV.jpg", width='700'><br><br>
</div>
