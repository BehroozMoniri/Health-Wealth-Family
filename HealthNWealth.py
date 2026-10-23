import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
df = pd.read_excel('data\Happiness Health and Economy of world.xlsx', sheet_name='2015')

features = ['Region', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
            'Freedom', 'Trust (Government Corruption)', 'Generosity']
X = df[features]
y = df['Happiness Score']

numeric_features = features[1:]
categorical_features = ['Region']

# Preprocessor to handle scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data ONCE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. TRAIN MODELS (Using Pipelines)
# ==========================================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}
trained_pipelines = {}

print("--- MODEL EVALUATION ---")
for name, model in models.items():
    # Create a pipeline for each model
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])

    # Cross-validation score
    scores = cross_val_score(clf, X, y, cv=5, scoring='r2')

    # Fit on training data for our plots
    clf.fit(X_train, y_train)
    trained_pipelines[name] = clf

    results[name] = {
        'Mean R2': scores.mean(),
        'Std R2': scores.std()
    }
    print(f"{name} R2: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ==========================================
# 3. PLOT 1: Actual vs. Predicted (Random Forest)
# ==========================================
print("\nGenerating Plots...")
rf_pipeline = trained_pipelines["Random Forest"]
y_pred_rf = rf_pipeline.predict(X_test)
actual_r2 = r2_score(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.7, color='blue', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title(f'Random Forest: Actual vs Predicted (Test R² = {actual_r2:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig('images/actual_vs_predicted.png', dpi=300)
plt.show()

# ==========================================
# 4. PLOT 2: Model Comparison Bar Chart
# ==========================================
# ==========================================
# PLOT 2: Model Comparison Bar Chart (Fixed)
# ==========================================
model_names = list(results.keys())
r2_scores = [results[m]['Mean R2'] for m in model_names]
errors = [results[m]['Std R2'] for m in model_names]

plt.figure(figsize=(10, 6))
# Use a distinct color for negative bars to highlight the problem
colors = ['#4C72B0' if score >= 0 else '#FF5733' for score in r2_scores]

bars = plt.bar(model_names, r2_scores, yerr=errors, capsize=5, color=colors)

# Add a horizontal line at 0 to make it clear
plt.axhline(0, color='black', linewidth=1)

plt.ylabel('Cross-Validated R² Score')
plt.title('Model Performance Comparison')

# DYNAMIC Y-AXIS: This ensures negative scores are visible
min_val = min(min(r2_scores) - 1, -1)
max_val = max(max(r2_scores) + 0.5, 1.0)
plt.ylim(min_val, max_val)

# Add text labels to the bars
for bar in bars:
    yval = bar.get_height()
    # Place text below the bar if it's negative, above if positive
    text_y = yval - 0.5 if yval < 0 else yval + 0.02
    plt.text(bar.get_x() + bar.get_width()/2, text_y, f"{yval:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('images/model_comparison_fixed.png', dpi=300)
plt.show()

# ==========================================
# 5. PLOT 3: Feature Importance (Random Forest)
# ==========================================
# Extract feature names after OneHotEncoding
preprocessor.fit(X_train)  # Fit to get the exact feature names
encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(encoded_cat_features)

# Get importances from the Random Forest model inside the pipeline
importances = rf_pipeline.named_steps['model'].feature_importances_

# Sort and get top 10
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.barh(range(len(indices)), importances[indices], color='#C44E52', align='center')
plt.yticks(range(len(indices)), [all_feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=300)
plt.show()

# ==========================================
# 6. PLOT 4: Correlation Heatmap
# ==========================================
plt.figure(figsize=(10, 8))
numeric_df = df[numeric_features + ['Happiness Score']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix of Happiness Metrics')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=300)
plt.show()

print("All plots generated and saved as PNG files.")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# 1. Convert Scores to Categories
def categorize_happiness(score):
    if score < 4.5:
        return 'Low'
    elif score <= 6.0:
        return 'Medium'
    else:
        return 'High'

df['Happiness Category'] = df['Happiness Score'].apply(categorize_happiness)
y_class = df['Happiness Category']

# 2. Use the same features (make sure to fill NaNs if there are any!)
X_class = df[['Region', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
              'Freedom', 'Trust (Government Corruption)', 'Generosity']].fillna(0)

# 3. Train Test Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# 4. We must use a CLASSIFIER now (RandomForestClassifier), not a Regressor
clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', RandomForestClassifier(n_estimators=100, random_state=42))])

clf_pipeline.fit(X_train_c, y_train_c)
y_pred_c = clf_pipeline.predict(X_test_c)

# 5. Print Accuracy
print(f"Classification Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
print("\nClassification Report:\n", classification_report(y_test_c, y_pred_c))

# 6. Plot Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c, labels=['Low', 'Medium', 'High'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix: Predicting Happiness Category')
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.show()

from sklearn.metrics import mean_absolute_error

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

print(f"{'Model':<20} | {'Train R²':<10} | {'Test R²':<10} | {'Test MAE':<10}")
print("-" * 55)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================================
# PLOT 1: Confusion Matrix Heatmap
# ==========================================
plt.figure(figsize=(8, 6))
# Assuming 'cm' was calculated in the previous step
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            annot_kws={"size": 16}) # Make numbers bigger
plt.title('Confusion Matrix: Predicting Happiness Category\n(Accuracy: 56%)', fontsize=14, pad=15)
plt.xlabel('Predicted Category', fontsize=12)
plt.ylabel('Actual Category', fontsize=12)
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=300)
plt.show()


# ==========================================
# PLOT 2: The Winner (Linear Regression) Predictions
# ==========================================
# Let's train just the winning Linear Regression model
winner_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', LinearRegression())])
winner_pipeline.fit(X_train, y_train)

# Get predictions for BOTH Train and Test sets
train_preds = winner_pipeline.predict(X_train)
test_preds = winner_pipeline.predict(X_test)

plt.figure(figsize=(10, 7))

# Plot Training Data
sns.scatterplot(x=y_train, y=train_preds, color='grey', alpha=0.5, label='Training Data')

# Plot Testing Data
sns.scatterplot(x=y_test, y=test_preds, color='blue', edgecolor='k', s=80, label='Testing Data')

# Plot Perfect Prediction Line
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')

plt.xlabel('Actual Happiness Score', fontsize=12)
plt.ylabel('Predicted Happiness Score', fontsize=12)
plt.title('Linear Regression: Train vs Test Predictions\n(Test R² = 0.864)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('images/winner_predictions.png', dpi=300)
plt.show()
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Train
    clf.fit(X_train, y_train)

    # Predict
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    # Evaluate
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    print(f"{name:<20} | {train_r2:<10.3f} | {test_r2:<10.3f} | {test_mae:<10.3f}")



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pandas as pd

# 1. Use ONLY the numerical features for clear interpretation
features_to_interpret = ['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
                         'Freedom', 'Trust (Government Corruption)', 'Generosity']

X_interpret = df[features_to_interpret].fillna(0)
y_interpret = df['Happiness Score']

# 2. Fit a simple Linear Regression
lr_interpret = LinearRegression()
lr_interpret.fit(X_interpret, y_interpret)

# 3. Create a DataFrame of the coefficients
coef_df = pd.DataFrame({
    'Feature': features_to_interpret,
    'Coefficient': lr_interpret.coef_
}).sort_values(by='Coefficient', ascending=False)

# 4. Plot the Coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, hue='Feature', palette='viridis', legend=False)
plt.title('The Drivers of Global Happiness (Linear Regression Coefficients)', fontsize=14)
plt.xlabel('Impact on Happiness Score (Coefficient)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig('images/happiness_coefficients.png', dpi=300)
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
df = pd.read_excel('data\Happiness Health and Economy of world.xlsx', sheet_name='2015')

# 2. Categorize Happiness
def categorize_happiness(score):
    if score < 4.5: return 'Low'
    elif score <= 6.0: return 'Medium'
    else: return 'High'

df['Actual Category'] = df['Happiness Score'].apply(categorize_happiness)

# 3. Setup Features
features = ['Region', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
            'Freedom', 'Trust (Government Corruption)', 'Generosity']
X = df[features]
y = df['Actual Category']

# 4. Split Data (We use random_state=42 to match your previous test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Preprocessor and Model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features[1:]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Region'])
    ])

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Train and Predict
clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)

# 7. Create a DataFrame of Results
# We grab the original Country names from the test set indices
results_df = pd.DataFrame({
    'Country': df.loc[y_test.index, 'Country'],
    'Actual Category': y_test,
    'Predicted Category': y_pred,
    'Actual Score': df.loc[y_test.index, 'Happiness Score']
})

# Add a column to show if the prediction was correct or wrong
results_df['Result'] = results_df.apply(lambda row: '✅ Correct' if row['Actual Category'] == row['Predicted Category'] else '❌ Incorrect', axis=1)

# Sort so the incorrect predictions are at the top (these are the most interesting to read!)
results_df = results_df.sort_values(by='Result', ascending=False)

# Save to CSV so we can read it
results_df.to_csv('classification_results.csv', index=False)
print("File saved as classification_results.csv")