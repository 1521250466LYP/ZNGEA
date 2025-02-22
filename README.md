# ZNGEA
ZNGEA：ZINB-NMF intergated Graph Embedding Autoencoder for metabolite-disease assiciation identification

##### Requirement

- numpy                     1.23.5
- pandas                    2.2.2
- python                    3.9.19
- scipy                     1.13.1
- torch                     2.2.1+cu121              
- scikit-learn              1.5.1

##### Dataset

- **Statistics**
    -Metabolites：2262
    -Diseases: 216
    -Validated metabolite-disease assiciations: 4536

##### File Introduction:

1. `bilineardecoder.py` Definition of Bilinear Decoder.
2. `data_processing.py` Processing the Adjacency Matrix. 
3. `Encoder_Decoder.py` ZINB-based Graph Convolutional Autoencoder. 
4. `GCN.py` The Convolution Process in a Graph Convolutional Network (GCN). 
5. `getsample.py` Obtain Samples.
6. `main.py` Trains ZNGEA Model.
7. `NMF.py` Non-negative Matrix Factorization.
8. `similarity_fusion.py` Similarity Fusion.
9. `ZINB.py` Loss Calculation of ZINB.
10. `M_D.csv` Metabolite-Disease Adjacency Matrix.
11. `d_fusion_sim.csv` Integrated Disease similarity Network.
12. `m_fusion_sim.zip` Integrated Metabolite similarity Network.
13. `metabolite_GIP_similarity.csv` Metabolite Gaussian Interaction Profile Kernel Similarity(MGIP).
14. `metabolites_information_entropy_similarity.csv` Metabolite Similarity based on Information Entropy (MSIE).
15. `metabolites_structure_similarity.zip` Metabolite Structural Similarity (MSS).
16. `disease_GIP_similarity.csv` Disease Gaussian Interaction Profile Kernel Similarity (DGIP).
17. `disease _information_entropy_similarity .csv` Disease Similarity based on Information Entropy (DSIE).
18. `disease_semantic_similarity.csv` Disease Semantic Similarity (DSS).
19. `disease name.xlsx` Disease Names Included in the Selected Dataset.
20. `metabolite name.xlsx` Metabolite Names Included in the Selected Dataset.
        
##### Running for ZNGEA

1. Unzip `m_fusion_sim.zip` and `metabolites_structure_similarity.zip` files.
2. Set up the environment based on the provided configuration.
3. Run the `main.py` file to start the model training and prediction.

##### Hyperparameter settings
- epochs_ZINB               60
- nhid1                     512
- nhid2                     64
- epoch_Decoder             10000
- dropout                   0.2              
- lr_ZINB                   0.0001
- weight_decay              5e-4
- decoder_dropout           0.4
- decoder_lr                0.0001



        

