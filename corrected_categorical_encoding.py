# Corrected Categorical Encoding Section

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def process_categorical_features(train_df, test_df, cat_cols):
    """
    Process categorical features consistently between train and test data
    """
    encoders = {}
    
    # Process training data
    for col in cat_cols:
        if col in train_df.columns:
            encoder = LabelEncoder()
            # Handle missing values consistently
            train_df[col] = train_df[col].fillna('missing').astype(str)
            train_df[col] = encoder.fit_transform(train_df[col])
            encoders[col] = encoder
    
    # Process test data with same encoders
    for col in cat_cols:
        if col in test_df.columns:
            # Handle missing values consistently
            test_df[col] = test_df[col].fillna('missing').astype(str)
            
            # Handle unseen categories
            def safe_transform(x):
                try:
                    return encoders[col].transform([x])[0]
                except ValueError:
                    # If category not seen in training, encode as 'missing'
                    return encoders[col].transform(['missing'])[0]
            
            test_df[col] = test_df[col].apply(safe_transform)
        else:
            # If column doesn't exist in test, fill with 'missing' encoding
            test_df[col] = encoders[col].transform(['missing'])[0]
    
    return train_df, test_df, encoders

# Corrected feature processing function
def process_features_consistently(df, is_train=True, encoders=None):
    """
    Process features consistently between train and test
    """
    # Convert object columns to numeric or category
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                df[col] = df[col].astype('category')
    
    # Handle missing values consistently
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
                
                # Handle unseen categories
                def safe_transform(x):
                    try:
                        return encoders[col].transform([str(x)])[0]
                    except ValueError:
                        return encoders[col].transform(['missing'])[0]
                
                df[col] = df[col].apply(safe_transform)
            else:
                df[col] = 0  # Default for missing features
        
        return df

# Usage example (to replace the problematic sections in your code):
"""
# After loading and basic preprocessing of train data:
train_processed, encoders = process_features_consistently(train, is_train=True)

# After loading test data:
test_processed = process_features_consistently(test, is_train=False, encoders=encoders)

# Ensure feature consistency
FEATURES = [col for col in train_processed.columns if col not in ['y', 'id2', 'id3']]
X_train = train_processed[FEATURES]
X_test = test_processed[FEATURES]

# Validate that all features are present
missing_in_test = set(FEATURES) - set(X_test.columns)
if missing_in_test:
    for col in missing_in_test:
        X_test[col] = 0  # or appropriate default
        
# Reorder columns to match
X_test = X_test[FEATURES]
"""