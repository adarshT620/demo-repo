import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def process_features_consistently(df, encoders=None, is_train=True):
    """
    Process features consistently between train and test data
    """
    # Convert object columns to numeric or category
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                df[col] = df[col].astype('category')
    
    # Handle missing values for numeric columns
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(-1)
    
    # Handle categorical columns
    cat_cols = [col for col in df.select_dtypes(include='category').columns 
                if col not in ['y']]
    
    if is_train:
        # For training data, fit encoders
        encoders = {}
        for col in cat_cols:
            encoder = LabelEncoder()
            df[col] = df[col].cat.add_categories(['missing'])
            df[col] = df[col].fillna('missing')
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
        return df, encoders
    else:
        # For test data, use existing encoders
        for col in cat_cols:
            if col in encoders:
                df[col] = df[col].cat.add_categories(['missing'])
                df[col] = df[col].fillna('missing')
                
                # Handle unseen categories safely
                def safe_transform(x):
                    try:
                        return encoders[col].transform([str(x)])[0]
                    except ValueError:
                        return encoders[col].transform(['missing'])[0]
                
                df[col] = df[col].apply(safe_transform)
            else:
                # If column doesn't exist in encoders, set to missing encoding
                df[col] = 0
        
        return df

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the training data
    """
    try:
        print("Loading training data...")
        train = pd.read_parquet(os.path.join(data_path, 'train_data.parquet'))
        print(f"Training data loaded: {train.shape}")
        
        # Convert object columns to category initially
        for col in train.select_dtypes(include='object'):
            train[col] = train[col].astype('category')
        
        # Get unique IDs for filtering
        id2_list = train['id2'].unique().tolist()
        id3_list = train['id3'].unique().tolist()
        
        # Load and process event data
        print("Processing event data...")
        add_event = pd.read_parquet(os.path.join(data_path, 'add_event.parquet'), 
                                   columns=['id2', 'id3', 'id6', 'id7'])
        event = add_event[add_event['id2'].isin(id3_list) & add_event['id3'].isin(id2_list)]
        event['id6'] = pd.to_datetime(event['id6'], errors='coerce')
        event['id7'] = pd.to_datetime(event['id7'], errors='coerce')
        event['has_clicked'] = event['id7'].notnull().astype('int8')
        event['click_delay'] = (event['id7'] - event['id6']).dt.total_seconds()
        event_agg = event.groupby(['id2', 'id3']).agg({
            'has_clicked': 'sum',
            'click_delay': 'mean',
            'id6': 'count'
        }).rename(columns={'id6': 'impression_count'}).reset_index()
        
        # Load and process transaction data
        print("Processing transaction data...")
        trans = pd.read_parquet(os.path.join(data_path, 'add_trans.parquet'), 
                               columns=['id2', 'id8', 'f367', 'f374'])
        trans = trans[trans['id8'].isin(id2_list) & trans['id2'].isin(id3_list)]
        trans = trans.rename(columns={'id8': 'id2', 'id2': 'id3'})
        trans['f367'] = pd.to_numeric(trans['f367'], errors='coerce')
        trans_agg = trans.groupby(['id2', 'id3']).agg({
            'f367': ['sum', 'mean', 'count'],
            'f374': pd.Series.nunique
        })
        trans_agg.columns = ['trans_sum', 'trans_mean', 'trans_count', 'trans_industry_nunique']
        trans_agg = trans_agg.reset_index()
        
        # Load offer data
        print("Processing offer data...")
        offer = pd.read_parquet(os.path.join(data_path, 'offer_metadata.parquet'))
        offer = offer[offer['id3'].isin(id3_list)]
        
        # Merge all data
        print("Merging datasets...")
        train = train.merge(event_agg, on=['id2', 'id3'], how='left')
        train = train.merge(trans_agg, on=['id2', 'id3'], how='left')
        train = train.merge(offer, on='id3', how='left')
        
        # Drop columns consistently
        COLS_TO_DROP = ['id1', 'id4', 'id5', '__index_level_0__']
        train.drop(columns=COLS_TO_DROP, inplace=True, errors='ignore')
        
        print(f"Final training data shape: {train.shape}")
        return train, event_agg, trans_agg, offer
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

def load_and_preprocess_test_data(data_path, event_agg, trans_agg, offer):
    """
    Load and preprocess test data consistently with training data
    """
    try:
        print("Loading test data...")
        test = pd.read_parquet(os.path.join(data_path, 'test_data.parquet'))
        print(f"Test data loaded: {test.shape}")
        
        # Save original IDs for submission
        original_test_ids = test[['id1', 'id2', 'id3', 'id5']].copy()
        
        # Convert object columns to category initially
        for col in test.select_dtypes(include='object'):
            test[col] = test[col].astype('category')
        
        # Merge additional data
        print("Merging additional data to test...")
        test = test.merge(event_agg, on=['id2', 'id3'], how='left')
        test = test.merge(trans_agg, on=['id2', 'id3'], how='left')
        test = test.merge(offer, on='id3', how='left')
        
        # Drop columns consistently
        COLS_TO_DROP = ['id1', 'id4', 'id5', '__index_level_0__']
        test.drop(columns=COLS_TO_DROP, inplace=True, errors='ignore')
        
        print(f"Final test data shape: {test.shape}")
        return test, original_test_ids
        
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        raise
    except Exception as e:
        print(f"Error processing test data: {e}")
        raise

# === MAIN EXECUTION ===

# Data path
data_path = "/kaggle/input/amex-challenge"

# === STEP 1: Load and Preprocess Full Train Data ===
print("=" * 50)
print("STEP 1: Loading and preprocessing training data")
print("=" * 50)

train, event_agg, trans_agg, offer = load_and_preprocess_data(data_path)

# Process features consistently
print("Processing training features...")
train_processed, encoders = process_features_consistently(train, is_train=True)

# Save processed training data
train_processed.to_csv("train_cleaned_full.csv", index=False)
print("✅ Processed training data saved to train_cleaned_full.csv")

# === STEP 2: Train Model with CV on ALL Features ===
print("\n" + "=" * 50)
print("STEP 2: Training model with cross-validation")
print("=" * 50)

TARGET = 'y'
ID_COLS = ['id2', 'id3']
FEATURES = [col for col in train_processed.columns if col not in [TARGET] + ID_COLS]

print(f"Number of features: {len(FEATURES)}")
print(f"Target distribution: {train_processed[TARGET].value_counts()}")

X = train_processed[FEATURES]
y = train_processed[TARGET].astype(int)

# Validate data
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Missing values in X: {X.isnull().sum().sum()}")

# Cross-validation training
importances = pd.Series(0, index=FEATURES)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = []
cv_scores = []

print("Starting cross-validation training...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/5")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.2,
        scale_pos_weight=5,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50), log_evaluation(0)]
    )
    
    # Validate fold
    val_pred = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_pred)
    cv_scores.append(auc_score)
    print(f"Fold {fold + 1} AUC: {auc_score:.4f}")
    
    models.append(model)
    importances += pd.Series(model.feature_importances_, index=FEATURES)

# Average importance across folds
importances /= 5
print(f"\nCross-validation AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Plot feature importance
plt.figure(figsize=(10, 6))
importances.sort_values(ascending=True).tail(50).plot(kind='barh')
plt.title("Top 50 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# Save models
joblib.dump(models, "lgb_models_cv_all_features.pkl")
joblib.dump(encoders, "feature_encoders.pkl")
print("✅ Models and encoders saved")

# === STEP 3: Test Predictions using ALL FEATURES ===
print("\n" + "=" * 50)
print("STEP 3: Making predictions on test data")
print("=" * 50)

test, original_test_ids = load_and_preprocess_test_data(data_path, event_agg, trans_agg, offer)

# Process test features consistently
print("Processing test features...")
test_processed = process_features_consistently(test, encoders=encoders, is_train=False)

# Ensure all features are present and in correct order
print("Validating feature consistency...")
for col in FEATURES:
    if col not in test_processed.columns:
        print(f"Adding missing feature: {col}")
        test_processed[col] = 0

# Reorder columns to match training
X_test = test_processed[FEATURES]

# Validate shapes
print(f"X_train shape: {X.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Feature count match: {X.shape[1] == X_test.shape[1]}")

if X.shape[1] != X_test.shape[1]:
    raise ValueError("Feature count mismatch between train and test!")

# Make predictions
print("Making predictions...")
models = joblib.load("lgb_models_cv_all_features.pkl")
test_predictions = np.zeros(len(X_test))

for i, model in enumerate(models):
    pred = model.predict_proba(X_test)[:, 1]
    test_predictions += pred
    print(f"Model {i+1}/5 predictions completed")

test_predictions /= len(models)

# Create submission
print("Creating submission file...")
submission = original_test_ids.copy()
submission['pred'] = test_predictions

# Apply MAP@7 ranking
submission['rank'] = submission.groupby('id2')['pred'].rank(method='first', ascending=False)
submission_final = submission[submission['rank'] <= 7].drop(columns='rank')

# Validate submission
print(f"Submission shape: {submission_final.shape}")
print(f"Unique id2 count: {submission_final['id2'].nunique()}")
print(f"Average predictions per id2: {submission_final.groupby('id2').size().mean():.2f}")

# Save submission
submission_final.to_csv("submission_all_features_ranked.csv", index=False)
print("✅ submission_all_features_ranked.csv is ready with MAP@7 format!")

print("\n" + "=" * 50)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("=" * 50)
print("Files created:")
print("- train_cleaned_full.csv")
print("- lgb_models_cv_all_features.pkl")
print("- feature_encoders.pkl")
print("- feature_importance.png")
print("- submission_all_features_ranked.csv")