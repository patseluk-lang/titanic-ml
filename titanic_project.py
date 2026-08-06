"""
=============================================================================
 TITANIC SURVIVAL PREDICTION - full ML project cycle (revised version)
=============================================================================
 1. Problem Definition          6. Model Selection
 2. Data Collection             7. Model Evaluation
 3. Data Cleaning               8. Fine-tuning & Validation
 4. EDA                         9. Deployment (model persistence)
 5. Feature Engineering        10. Documentation

 Revisions:
   * the dataset is cached locally after the first download
   * Age: refined logic - group median first, then global median
   * Fare=0 is handled correctly on the log-scaled plot
   * FareBin is built safely (avoids the qcut duplicate-edges error)
   * feature importance is guarded by an attribute check
   * LogReg / SVM use class_weight='balanced' due to class imbalance
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             RocCurveDisplay)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style='whitegrid')

# Output folder - next to the script, created automatically
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


# =============================================================================
# 1. PROBLEM DEFINITION
# =============================================================================
# Task: binary classification. Predict Survived (0/1) from the passenger's
# demographic and ticket features.
# Metric: accuracy (primary) + F1/ROC-AUC (due to class imbalance ~62/38).
print("=" * 70)
print("1. PROBLEM DEFINITION: binary classification of Survived (0/1)")
print("=" * 70)


# =============================================================================
# 2. DATA COLLECTION
# =============================================================================
URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
RAW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titanic_raw.csv')

if os.path.exists(RAW_PATH):
    df = pd.read_csv(RAW_PATH)
    print("   Loaded the local copy titanic_raw.csv")
else:
    df = pd.read_csv(URL)
    df.to_csv(RAW_PATH, index=False)
    print("   Downloaded from the internet and cached locally as titanic_raw.csv")

print(f"\n2. DATA COLLECTION")
print(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"   Columns: {df.columns.tolist()}")
print(f"\n   First rows:\n{df.head(3).to_string()}")
print(f"\n   Types and missing values:")
info = pd.DataFrame({
    'dtype': df.dtypes,
    'nulls': df.isnull().sum(),
    'null_%': (df.isnull().sum() / len(df) * 100).round(1),
    'unique': df.nunique()
})
print(info.to_string())


# =============================================================================
# 3. DATA CLEANING AND PREPROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("3. DATA CLEANING")
print("=" * 70)

df_clean = df.copy()

# --- Title from Name (before dropping Name) ---
df_clean['Title'] = df_clean['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
title_map = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Ms': 'Miss', 'Mlle': 'Miss', 'Mme': 'Mrs',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Capt': 'Rare', 'Lady': 'Rare', 'Sir': 'Rare', 'Don': 'Rare',
    'Dona': 'Rare', 'the Countess': 'Rare', 'Jonkheer': 'Rare'
}
df_clean['Title'] = df_clean['Title'].map(title_map).fillna('Rare')
print(f"\n   Title (from Name):\n{df_clean['Title'].value_counts().to_string()}")

# --- Age: median within the group (Title + Pclass);
#     if every value in a group is NaN (no median), fall back to the global median ---
before = df_clean['Age'].isnull().sum()
df_clean['Age'] = df_clean.groupby(['Title', 'Pclass'])['Age'].transform(
    lambda s: s.fillna(s.median()))
# Safety net: if some (Title, Pclass) combination held only NaN, groupby.transform
# returns NaN - those are then filled with the global median.
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
print(f"\n   Age: filled {before} missing values with the group median (Title x Pclass); "
      f"leftovers with the global median")

# --- Embarked: mode (only 2 missing values) ---
before = df_clean['Embarked'].isnull().sum()
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
print(f"   Embarked: filled {before} missing values with the mode ('{df_clean['Embarked'].mode()[0]}')")

# --- Fare: median per Pclass ---
df_clean['Fare'] = df_clean.groupby('Pclass')['Fare'].transform(
    lambda s: s.fillna(s.median()))

# --- Cabin: 77% missing -> not imputed; the deck and a flag are extracted instead ---
df_clean['Deck'] = df_clean['Cabin'].str[0].fillna('U')          # U = Unknown
df_clean['Deck'] = df_clean['Deck'].replace('T', 'U')            # 'T' - 1 passenger, no signal
df_clean['HasCabin'] = df_clean['Cabin'].notnull().astype(int)
print(f"   Cabin: 77% missing -> replaced by Deck (first letter) + HasCabin (0/1)")

# --- Drop the features that carry no signal ---
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_clean = df_clean.drop(columns=drop_cols)
print(f"   Dropped: {drop_cols} (Name -> Title, Cabin -> Deck already extracted)")
print(f"\n   Missing values left: {df_clean.isnull().sum().sum()}")


# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("4. EDA")
print("=" * 70)

print(f"\n   Class balance:\n{df_clean['Survived'].value_counts(normalize=True).round(3).to_string()}")

for col in ['Sex', 'Pclass', 'Embarked', 'Title']:
    rate = df_clean.groupby(col)['Survived'].agg(['mean', 'count']).round(3)
    print(f"\n   Survival by {col}:\n{rate.to_string()}")

print(f"\n   Numeric features:\n{df_clean[['Age', 'Fare', 'SibSp', 'Parch']].describe().round(2).to_string()}")

# --- EDA plots ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('EDA: Titanic Survival', fontsize=16, fontweight='bold')

sns.countplot(data=df_clean, x='Survived', hue='Survived', legend=False, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Target variable distribution')

sns.barplot(data=df_clean, x='Sex', y='Survived', hue='Sex', legend=False, ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Survival by sex')

sns.barplot(data=df_clean, x='Pclass', y='Survived', hue='Pclass', legend=False, ax=axes[0, 2], palette='Set2')
axes[0, 2].set_title('Survival by class')

sns.histplot(data=df_clean, x='Age', hue='Survived', bins=30, kde=True,
             ax=axes[1, 0], palette='Set1', multiple='layer', alpha=0.6)
axes[1, 0].set_title('Age distribution')

# For the log scale, Fare=0 is replaced with 0.01 to avoid -inf
plot_df = df_clean.copy()
plot_df['Fare_plot'] = plot_df['Fare'].replace(0, 0.01)
sns.boxplot(data=plot_df, x='Survived', y='Fare_plot', hue='Survived', legend=False, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('Fare (log) vs Survived')
axes[1, 1].set_ylabel('Fare')

sns.barplot(data=df_clean, x='Title', y='Survived', hue='Title', legend=False, ax=axes[1, 2], palette='Set2')
axes[1, 2].set_title('Survival by title')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'eda_overview.png'), dpi=110, bbox_inches='tight')
plt.close()

# --- Correlation matrix ---
num_df = df_clean.copy()
num_df['Sex_n'] = (num_df['Sex'] == 'male').astype(int)
corr = num_df[['Survived', 'Pclass', 'Sex_n', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin']].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
plt.title('Correlation matrix', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'correlation.png'), dpi=110, bbox_inches='tight')
plt.close()

print(f"\n   Correlation with Survived:\n{corr['Survived'].drop('Survived').sort_values(key=abs, ascending=False).round(3).to_string()}")
print("\n   Plots: eda_overview.png, correlation.png")


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("5. FEATURE ENGINEERING")
print("=" * 70)

df_feat = df_clean.copy()

df_feat['FamilySize'] = df_feat['SibSp'] + df_feat['Parch'] + 1
df_feat['IsAlone'] = (df_feat['FamilySize'] == 1).astype(int)

AGE_BINS = [0, 12, 20, 40, 60, 100]
AGE_LABELS = ['Child', 'Teen', 'Adult', 'Middle', 'Senior']
df_feat['AgeBin'] = pd.cut(df_feat['Age'], bins=AGE_BINS, labels=AGE_LABELS)

# Safe FareBin construction via quantiles (avoids the qcut duplicate-edges error)
fare_quantiles = df_feat['Fare'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
fare_bins_unique = np.unique(fare_quantiles)
if len(fare_bins_unique) < 2:
    fare_bins_unique = [df_feat['Fare'].min(), df_feat['Fare'].max()]
n_bins = len(fare_bins_unique) - 1
FARE_LABELS = ['Low', 'Mid', 'High', 'VeryHigh']
# If the quartiles produced fewer bins, the label list is trimmed accordingly
fare_labels_dynamic = FARE_LABELS[:n_bins] if n_bins <= len(FARE_LABELS) else [f'Q{i+1}' for i in range(n_bins)]
df_feat['FareBin'] = pd.cut(df_feat['Fare'], bins=fare_bins_unique, labels=fare_labels_dynamic, include_lowest=True)
FARE_BINS = np.array(fare_bins_unique, dtype=float)
FARE_BINS[0], FARE_BINS[-1] = -np.inf, np.inf  # open edges for unseen values

df_feat['FarePerPerson'] = df_feat['Fare'] / df_feat['FamilySize']

print("\n   New features: FamilySize, IsAlone, AgeBin, FareBin, FarePerPerson")
print(f"\n   Survival by FamilySize:\n{df_feat.groupby('FamilySize')['Survived'].agg(['mean', 'count']).round(3).to_string()}")
print(f"\n   Survival by IsAlone:\n{df_feat.groupby('IsAlone')['Survived'].agg(['mean', 'count']).round(3).to_string()}")

X = df_feat.drop(columns='Survived')
y = df_feat['Survived']

num_features = ['Age', 'Fare', 'FamilySize', 'FarePerPerson', 'SibSp', 'Parch']
cat_features = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeBin', 'FareBin']
pass_features = ['Pclass', 'IsAlone', 'HasCabin']

# Preprocessor: scaling of numeric features (for SVM/KNN/LogReg) + OHE of categoricals
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler())]), num_features),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))]), cat_features),
    ('pass', 'passthrough', pass_features),
])


# =============================================================================
# 6. MODEL SELECTION
# =============================================================================
print("\n" + "=" * 70)
print("6. MODEL SELECTION")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"\n   Train: {X_train.shape[0]} | Test: {X_test.shape[0]} (stratify=y)")

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
    'KNN':                KNeighborsClassifier(n_neighbors=7),
    'SVM (RBF)':          SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
    'DecisionTree':       DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    'RandomForest':       RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    'GradientBoosting':   GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = []

for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_results.append({'Model': name, 'CV_mean': scores.mean(), 'CV_std': scores.std()})
    print(f"   {name:20s} CV accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

cv_df = pd.DataFrame(cv_results).sort_values('CV_mean', ascending=False).reset_index(drop=True)
print(f"\n   CV ranking:\n{cv_df.round(4).to_string(index=False)}")


# =============================================================================
# 7. MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("7. MODEL EVALUATION (on the test set)")
print("=" * 70)

eval_results = []
fitted = {}

for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', model)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    eval_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'ROC_AUC': roc_auc_score(y_test, proba),
    })

eval_df = pd.DataFrame(eval_results).sort_values('F1', ascending=False).reset_index(drop=True)
print(f"\n{eval_df.round(4).to_string(index=False)}")

# --- Confusion matrices for every model ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Confusion Matrices (test set)', fontsize=15, fontweight='bold')
for ax, (name, pipe) in zip(axes.ravel(), fitted.items()):
    cm = confusion_matrix(y_test, pipe.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
    ax.set_title(name)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'confusion_matrices.png'), dpi=110, bbox_inches='tight')
plt.close()

# --- ROC curves ---
plt.figure(figsize=(8, 7))
ax = plt.gca()
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.title('ROC Curves', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'roc_curves.png'), dpi=110, bbox_inches='tight')
plt.close()
print("\n   Plots: confusion_matrices.png, roc_curves.png")


# =============================================================================
# 8. FINE-TUNING AND VALIDATION
# =============================================================================
print("\n" + "=" * 70)
print("8. FINE-TUNING (GridSearchCV)")
print("=" * 70)

print(f"\n   CV leader: {cv_df.iloc[0]['Model']}")
print("   Tuning two ensembles (RandomForest, GradientBoosting); the final pick is the better CV score")

grids = {
    'RandomForest': (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {'clf__n_estimators': [200, 400],
         'clf__max_depth': [5, 8, None],
         'clf__min_samples_split': [2, 5, 10],
         'clf__min_samples_leaf': [1, 2, 4],
         'clf__max_features': ['sqrt', 'log2']}
    ),
    'GradientBoosting': (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {'clf__n_estimators': [100, 200, 300],
         'clf__learning_rate': [0.05, 0.1],
         'clf__max_depth': [2, 3, 4],
         'clf__subsample': [0.8, 1.0]}
    ),
}

tuned = {}
for name, (model, grid) in grids.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', model)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    tuned[name] = gs
    print(f"\n   {name}")
    print(f"      best CV accuracy: {gs.best_score_:.4f}")
    print(f"      best params: { {k.replace('clf__',''): v for k, v in gs.best_params_.items()} }")

# --- Final model ---
final_name = max(tuned, key=lambda k: tuned[k].best_score_)
final_model = tuned[final_name].best_estimator_
print(f"\n   >>> FINAL MODEL: {final_name} (CV={tuned[final_name].best_score_:.4f})")

y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

print(f"\n   VALIDATION ON THE TEST SET:")
print(f"      Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"      Precision: {precision_score(y_test, y_pred):.4f}")
print(f"      Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"      F1       : {f1_score(y_test, y_pred):.4f}")
print(f"      ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Died', 'Survived'])}")

cm = confusion_matrix(y_test, y_pred)
print(f"   Confusion matrix:\n      TN={cm[0,0]}  FP={cm[0,1]}\n      FN={cm[1,0]}  TP={cm[1,1]}")

# --- Feature importance (guarded by an attribute check) ---
clf = final_model.named_steps['clf']
feat_names = final_model.named_steps['prep'].get_feature_names_out()

if hasattr(clf, 'feature_importances_'):
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}) \
               .sort_values('importance', ascending=False).head(15)
    print(f"\n   TOP-15 most important features:\n{imp_df.round(4).to_string(index=False)}")

    plt.figure(figsize=(9, 7))
    sns.barplot(data=imp_df, y='feature', x='importance', hue='feature', legend=False, palette='viridis')
    plt.title(f'Feature Importance - {final_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'feature_importance.png'), dpi=110, bbox_inches='tight')
    plt.close()
    print("   Plot: feature_importance.png")
elif hasattr(clf, 'coef_'):
    importances = np.abs(clf.coef_[0])
    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}) \
               .sort_values('importance', ascending=False).head(15)
    print(f"\n   TOP-15 weight coefficients (|coef|):\n{imp_df.round(4).to_string(index=False)}")

    plt.figure(figsize=(9, 7))
    sns.barplot(data=imp_df, y='feature', x='importance', hue='feature', legend=False, palette='viridis')
    plt.title(f'Feature Weights (|coef|) - {final_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'feature_importance.png'), dpi=110, bbox_inches='tight')
    plt.close()
    print("   Plot: feature_importance.png")
else:
    print("\n   Feature importance is not available for this model.")


# =============================================================================
# 9. DEPLOYMENT
# =============================================================================
print("\n" + "=" * 70)
print("9. DEPLOYMENT")
print("=" * 70)

joblib.dump(final_model, os.path.join(OUT, 'titanic_model.pkl'))
print("\n   Model saved: titanic_model.pkl")


def predict_passenger(model, **kwargs):
    """Predict for a single passenger. Takes raw fields and derives the rest itself."""
    d = {
        'Pclass': kwargs['Pclass'], 'Sex': kwargs['Sex'], 'Age': kwargs['Age'],
        'SibSp': kwargs.get('SibSp', 0), 'Parch': kwargs.get('Parch', 0),
        'Fare': kwargs['Fare'], 'Embarked': kwargs.get('Embarked', 'S'),
        'Title': kwargs.get('Title', 'Mr'), 'Deck': kwargs.get('Deck', 'U'),
        'HasCabin': kwargs.get('HasCabin', 0),
    }
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
    d['IsAlone'] = int(d['FamilySize'] == 1)
    d['FarePerPerson'] = d['Fare'] / d['FamilySize']
    d['AgeBin'] = pd.cut([d['Age']], bins=AGE_BINS, labels=AGE_LABELS)[0]
    d['FareBin'] = pd.cut([d['Fare']], bins=FARE_BINS, labels=FARE_LABELS[:len(FARE_BINS)-1])[0]
    row = pd.DataFrame([d])
    p = model.predict_proba(row)[0, 1]
    return {'survived': int(p >= 0.5), 'probability': round(float(p), 4)}


demos = [
    dict(name='Woman, 1st class, age 30', Pclass=1, Sex='female', Age=30, Fare=100, Title='Mrs', Deck='C', HasCabin=1, SibSp=1),
    dict(name='Man, 3rd class, age 25',   Pclass=3, Sex='male',   Age=25, Fare=7.5, Title='Mr'),
    dict(name='Boy, 2nd class, age 8',    Pclass=2, Sex='male',   Age=8,  Fare=26,  Title='Master', SibSp=1, Parch=2),
]
print("\n   Demo predictions:")
for d in demos:
    label = d.pop('name')
    r = predict_passenger(final_model, **d)
    status = 'SURVIVES' if r['survived'] else 'DOES NOT SURVIVE'
    print(f"      {label:28s} -> {status:16s} (p={r['probability']})")


# =============================================================================
# 10. DOCUMENTATION AND PRESENTATION
# =============================================================================
print("\n" + "=" * 70)
print("10. SUMMARY")
print("=" * 70)
print(f"""
   Data:        891 passengers, 12 source columns + derived features after engineering
   Missing:     Age (median TitlexPclass, then global), Embarked (mode), Cabin -> Deck+HasCabin
   Features:    Title, FamilySize, IsAlone, AgeBin, FareBin, FarePerPerson, Deck
   Models:      6 algorithms, 5-fold Stratified CV
   Final:       {final_name}
   Test:        Accuracy={accuracy_score(y_test, y_pred):.4f} | F1={f1_score(y_test, y_pred):.4f} | ROC-AUC={roc_auc_score(y_test, y_proba):.4f}

   Key findings:
     * Sex is the strongest predictor (women ~74% vs men ~19%)
     * Ticket class: 1st ~63%, 2nd ~47%, 3rd ~24%
     * Title (Master/Mrs/Miss) reinforces the sex and age signal
     * Optimal FamilySize = 2-4; solo travellers and large families fared worse
     * Having a cabin (HasCabin) correlates with wealth -> higher chances
""")

# Summary table of results
eval_df.round(4).to_csv(os.path.join(OUT, 'model_comparison.csv'), index=False)
print("   Comparison table: model_comparison.csv")
print("=" * 70)