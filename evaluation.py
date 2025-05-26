import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the uploaded Excel file to inspect the data and understand the structure
#file_path = './2_shalaka_diarization/a_gpt_re_4o.xlsx'
file_path = './657_samples/gemini_2_a4_total.xlsx'
#file_path = './1_google_diarization/a_google_re_35.xlsx'
print(file_path)

data = pd.read_excel(file_path)

# Display the first few rows to understand the columns and structure
data.head()

# Extract the subject prefix from the 'id' column (up to the second underscore)
data['subject'] = data['id'].str.extract(r'(^\w+_\d{6})')

#sum19_strong 
# Group by the extracted subject and sum the values in column 'sum19_strong'
subject_sum = data.groupby('subject')['sum'].sum().reset_index()

# Group by the extracted subject and calculate the average of values in column 'sum19_strong'
subject_avg = data.groupby('subject')['sum'].mean().reset_index()

# Rank the subjects based on the average of column '1' in descending order
subject_avg['Rank'] = subject_avg['sum'].rank(method='dense', ascending=False).astype(int)

# Sort by the Rank for easier readability
subject_avg = subject_avg.sort_values(by='Rank').reset_index(drop=True)

# Predict each record's label based on conditions provided
#Label = 1 if column F_1 > 0 or F_9 > 0 or sum > 1, otherwise label = 0
data['Pred'] = ((data['1'] > 0) | (data['9'] > 0) | (data['sum'] >= 1)).astype(int)

# Display results with labels for each record
data_with_labels = data[['id', '1', '9', 'sum', 'Pred','A4']]

# Calculate performance metrics
# Assuming 'A4' is the ground truth label column
accuracy = accuracy_score(data['A4'], data['Pred'])
precision = precision_score(data['A4'], data['Pred'])
recall = recall_score(data['A4'], data['Pred'])
f1 = f1_score(data['A4'], data['Pred'])

# Calculate confusion matrix
#tn, fp, fn, tp = confusion_matrix(data['A4'], data['Pred']).ravel()

# Calculate Specificity and PPV
#specificity = tn / (tn + fp)

# Print the calculated performance metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score : {f1:.4f}')
