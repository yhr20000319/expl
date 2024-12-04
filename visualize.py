import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('train_feature_importance.csv')
df2 = pd.read_csv('test_feature_importance.csv')
#df = pd.read_csv('train_feature_importance_iene.csv')
#df2 = pd.read_csv('test_feature_importance_iene.csv')

features = df['Feature']
importances = df['Importance']
importances2 = df2['Importance']
#features = df['Feature'][-100:]
#importances = df['Importance'][-100:]

def normalize(importances):
    return (importances - np.min(importances)) / (np.max(importances) - np.min(importances))

normalized_importances1 = normalize(importances)
normalized_importances2 = normalize(importances2)
x = range(len(features))

plt.figure(figsize=(10, 6))
width = 0.35

plt.bar([i - width/2 for i in x], normalized_importances1, color='skyblue', label='Group 1')


plt.bar([i - width/2 for i in x], normalized_importances2, color='salmon', label='Group 2', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


top100_indices1 = normalized_importances1.argsort()[-100:][::-1]
top100_indices2 = normalized_importances2.argsort()[-100:][::-1]


top100_features1 = features[top100_indices1]
top100_features2 = features[top100_indices2]


overlap_count = len(set(top100_features1) & set(top100_features2))
overlap_rate = overlap_count / 100

print(overlap_rate)


top100_indices1 = importances.argsort()[-200:][::-1]
top100_indices2 = importances2.argsort()[-200:][::-1]


last10_indices = range(len(features) - 10, len(features))


last10_count1 = sum(idx in last10_indices for idx in top100_indices1)
last10_count2 = sum(idx in last10_indices for idx in top100_indices2)

last10_ratio1 = last10_count1 / 10
last10_ratio2 = last10_count2 / 10

print(f"Group 1: Last 10 features in top 100 importance ratio: {last10_ratio1 * 100:.2f}%")
print(f"Group 2: Last 10 features in top 100 importance ratio: {last10_ratio2 * 100:.2f}%")