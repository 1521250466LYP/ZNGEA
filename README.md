# ZNGEA：ZINB-NMF intergated Graph Embedding Autoencoder for metabolite-disease assiciation identification

# If you want to run this model, you first need to unzip the `m_fusion_sim.zip` and `metabolites_structure_similarity.zip` files, and then run the `main.py` file.

# requirement

numpy                     1.23.5
pandas                    2.2.2
python                    3.9.19
scipy                     1.13.1
torch                     2.2.1+cu121              
scikit-learn              1.5.1

# File Introduction:
1. `bilineardecoder.py` Definition of Bilinear Decoder.
2. `data_processing.py` Processing the Adjacency Matrix. 
3. `Encoder_Decoder.py` ZINB-based graph convolutional autoencoder. 
4. `GCN.py` The convolution process in a Graph Convolutional Network (GCN). 
5. `getsample.py` Obtain samples.
6. `main.py` trains ZNGEA model.
7. `NMF.py` Non-negative Matrix Factorization.
8. `similarity_fusion.py` Similarity Fusion.
9. `ZINB.py` Loss Calculation of ZINB.
10. `M_D.csv` Metabolite-Disease Adjacency Matrix.
        

