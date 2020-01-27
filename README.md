# Brain-Region-Model-Evaluations

***Source***

"Sex Difference in the brain: Divergent results from traditional machine learning and convolutional networks"


***Purpose***

This research was conducted for the IEEE International Symposium on Biomedical Imaging (ISBI). ISBI is a scientific conference dedicated to mathematical, algorithmic, and computational aspects of biological and biomedical imaging, across all scales of observation. It fosters knowledge transfer among different imaging communities and contributes to an integrative approach to biomedical imaging. ISBI is a joint initiative from the IEEE Signal Processing Society (SPS) and the IEEE Engineering in Medicine and Biology Society (EMBS). 


***Abstract***

Neuroimaging research has begun adopting deep learning to model structural differences in the brain. This is a break from previous approaches that rely largely on anatomical volumetric or thickness-based features. Currently, most studies employ either convolutional deep learning based models or traditional machine learning models that use volumetric features. Because of this split, it is unclear which approach yields better predictive performance, or whether the two approaches will lead to different neuroanatomical conclusions, potentially even when applied to the same dataset. To address these questions, we present the largest single study of sex differences in the brain using 21,390 UK Biobank T1-weighted brain MRIs, which we analyzed through both traditional volumetric and 3D convolutional neural network models. Overall, we find that 3D-CNNs outperformed traditional machine learning models, with sex classification area under the ROC curve of 0.849 and 0.683, respectively. When performing sex classification using only single regions of the brain, we observed better performance from 3D-CNNs in all regions tested, indicating sex differences in the brain likely represent both structural and volumetric changes. In addition, we find little consensus in terms of brain region prioritization between the two approaches. In summary, we find that 3D-CNNs show exceptional sex classification performance, extract additional relevant structural information from brain regions beyond volume, and possibly because of this, prioritize sex differences in neuroanatomical regions differently than volume-based approaches. 