# CS8321_FinalLab

* [First Draft Google Doc](https://docs.google.com/document/d/11tZkmYneFWWXYQxjDA_bsvjTjB7J_KGn3i37CjPZKFY/edit?usp=sharing)
* [Final Paper Google Doc](https://docs.google.com/document/d/16gqENeRNzuZr_eyEJzhWJBORatIGVhtdW8yIHX_AqZI/edit?usp=sharing)
* [Miro Board](https://miro.com/app/board/uXjVNhYbgqA=/?share_link_id=369794203799)

# Verification Dataset

The dataset for verification is compiled from [LFWA+](https://liuziwei7.github.io/projects/FaceAttributes.html)
  * [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ3NzdzRhVUhSams?resourcekey=0-Kpdd6Vctf-AdJYfS55VULA)
  * Parsing of the data and pairing for verification protocol: Notebooks under [LFWA+_parsing_and_cleanup.ipynb/Pairing.ipynb](code/BuildingVerificationDataset)
  * ❗❗❗ To Install compiled dataset: [SMU Outlook One Drive](https://smu365-my.sharepoint.com/:f:/r/personal/galatoum_smu_edu/Documents/CS8321_Final_Lab?csf=1&web=1&e=O07wxP)❗❗❗
    
# Training Dataset 

* FairFace Dataset: 
    * [Github](https://github.com/joojs/fairface)
        * [Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf)
        * Used `Padding=0.25`
            * [GoogleDrive Images](https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view)
            * [GoogleDrive Labels](https://drive.google.com/file/d/1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH/view)
            
# ResNet50 Model Trained on VGGFace2 Dataset

This model was just utilized for purposes of confirming bias calculation methods.
* [Github Code](https://github.com/WeidiXie/Keras-VGGFace2-ResNet50/tree/69a608a2a140b7025bcb69adcd2355e38cc89f1d)
* [Weights Installed From](https://drive.google.com/file/d/1AHVpuB24lKAqNyRRjhX7ABlEor6ByZlS/view)


