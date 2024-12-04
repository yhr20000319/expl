# Supplementary materials of IENE: interpretability experiments

Table 1: Proportion of pseudo correlation features in the top 200 important features

|      | Proportion |
| ---- | ---------- |
| ERM  | 50%    |
| IENE | 0%     |

Here, we present an example of a toy to demonstrate that IENE has extracted invariant features. It uses Sensitivity Analysis (SA) [1] to illustrate the importance of features. We applied it separately to the ERM trained model and the IENE trained model to demonstrate that the former's performance degradation was due to the incorrect utilization of pseudo correlated features, while the latter successfully learned how to utilize invariant features. The following content will be added to the Appendix.

First, we use ERM to train a GNN model on ood Cora, and use SA method to measure the importance score of each feature to the model. Then, we evaluated the feature importance of training data and test data, and found that the coincidence rate of the top 100 important features of training data and test data was **<u>52%</u>**.

Although there are pseudo correlation features in the original 1433 features, they are difficult to measure. Therefore, in ood Cora, we have 10 manually generated pseudo correlation features stitched at the end, so we also measured the content of these pseudo correlation features in the top 200 important features. We find that the model uses **<u> 50%</u>** of the generated pseudo correlation features.

Then, we also use iene to train a GNN model on ood Cora, and use the same way to evaluate the importance of features. It is found that the coincidence rate of the top 100 important features of training data and test data is <font class="text-color-1" color="#f44336">**<u> 59% (+7%</u>)**</font> . This shows that although there is a shift in the distribution of training data and test data, the model still captures more invariant features for prediction.

In addition, we also measured the feature importance on the GNN trained by iene. Among the top 200 important features, the content of these pseudo correlation features is <font class="text-color-01" color="#f44336">**<u>0% (-50%)</u>**</font>. This shows that iene has successfully got rid of the false correlation.


We also presented the feature importance distribution maps of the models trained by two methods. 

![](https://oss-liuchengtu.hudunsoft.com/userimg/3f/3fb783477328979c276a3a38829e2786.png)

Fig1. the feature importance distribution maps of the models trained by ERM.

![](https://oss-liuchengtu.hudunsoft.com/userimg/d2/d2b7f6b431e01a48dd84e4311edd8085.png)

Fig2. the feature importance distribution maps of the models trained by IENE.



[1] Explainability techniques for graph convolutional networks, in International Conference on Machine Learning (ICML) Workshops, 2019.
