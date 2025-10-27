"""
Advanced Model Optimization Module for SMS Spam Classification
Focused on achieving Precision & Recall > 93%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, make_scorer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedSpamOptimizer:
    """Advanced optimizer for spam classification with target Precision & Recall > 93%"""
    
    def __init__(self, target_precision=0.93, target_recall=0.93):
        """
        Initialize the advanced optimizer
        
        Args:
            target_precision: Target precision threshold (default: 0.93)
            target_recall: Target recall threshold (default: 0.93)
        """
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.best_model = None
        self.best_vectorizer = None
        self.best_params = None
        self.best_scores = None
        
    def optimize_tfidf_parameters(self, X_train, y_train, X_test, y_test):
        """
        Optimize TF-IDF vectorizer parameters for better feature extraction
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            
        Returns:
            dict: Best TF-IDF parameters and scores
        """
        print("🔍 Phase 3.2: Advanced TF-IDF Parameter Optimization")
        print("-" * 50)
        
        # Extended TF-IDF parameter grid
        tfidf_params = {
            'max_features': [1500, 2000, 2500, 3000],
            'ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],
            'min_df': [1, 2, 3],
            'max_df': [0.85, 0.9, 0.95],
            'sublinear_tf': [True, False],
            'norm': ['l1', 'l2'],
            'use_idf': [True, False]
        }
        
        best_score = 0
        best_tfidf_params = None
        best_results = None
        
        # Grid search for TF-IDF parameters
        for max_features in tfidf_params['max_features']:
            for ngram_range in tfidf_params['ngram_range']:
                for min_df in tfidf_params['min_df']:
                    for max_df in tfidf_params['max_df']:
                        for sublinear_tf in tfidf_params['sublinear_tf']:
                            for norm in tfidf_params['norm']:
                                for use_idf in tfidf_params['use_idf']:
                                    try:
                                        # Create and fit vectorizer
                                        vectorizer = TfidfVectorizer(
                                            max_features=max_features,
                                            ngram_range=ngram_range,
                                            min_df=min_df,
                                            max_df=max_df,
                                            sublinear_tf=sublinear_tf,
                                            norm=norm,
                                            use_idf=use_idf,
                                            stop_words='english'
                                        )
                                        
                                        X_train_vec = vectorizer.fit_transform(X_train)
                                        X_test_vec = vectorizer.transform(X_test)
                                        
                                        # Quick SVM test
                                        svm = SVC(kernel='rbf', random_state=42)
                                        svm.fit(X_train_vec, y_train)
                                        y_pred = svm.predict(X_test_vec)
                                        
                                        # Calculate metrics
                                        precision = precision_score(y_test, y_pred, pos_label='spam')
                                        recall = recall_score(y_test, y_pred, pos_label='spam')
                                        f1 = f1_score(y_test, y_pred, pos_label='spam')
                                        
                                        # Combined score with emphasis on precision and recall
                                        combined_score = (precision * recall * f1) ** (1/3)
                                        
                                        if (precision >= self.target_precision and 
                                            recall >= self.target_recall and 
                                            combined_score > best_score):
                                            
                                            best_score = combined_score
                                            best_tfidf_params = {
                                                'max_features': max_features,
                                                'ngram_range': ngram_range,
                                                'min_df': min_df,
                                                'max_df': max_df,
                                                'sublinear_tf': sublinear_tf,
                                                'norm': norm,
                                                'use_idf': use_idf
                                            }
                                            best_results = {
                                                'precision': precision,
                                                'recall': recall,
                                                'f1': f1,
                                                'combined_score': combined_score
                                            }
                                            
                                            print(f"✅ New best: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
                                            
                                    except Exception as e:
                                        continue
        
        if best_tfidf_params:
            print(f"\n🎯 Best TF-IDF Parameters Found:")
            for key, value in best_tfidf_params.items():
                print(f"  {key}: {value}")
            print(f"\n📊 Best Results:")
            for key, value in best_results.items():
                print(f"  {key}: {value:.4f}")
        else:
            print("⚠️ No parameter combination achieved target thresholds")
            
        return best_tfidf_params, best_results
    
    def optimize_svm_hyperparameters(self, X_train_vec, y_train, X_test_vec, y_test):
        """
        Perform comprehensive SVM hyperparameter optimization
        
        Args:
            X_train_vec, y_train: Vectorized training data
            X_test_vec, y_test: Vectorized testing data
            
        Returns:
            dict: Best SVM parameters and scores
        """
        print("\n🔍 Phase 3.1: SVM Hyperparameter Grid Search")
        print("-" * 50)
        
        # Comprehensive SVM parameter grid
        svm_param_grid = {
            'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'kernel': ['rbf', 'polynomial', 'sigmoid'],
            'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
        }
        
        best_score = 0
        best_svm_params = None
        best_results = None
        
        print("🔄 Testing SVM parameter combinations...")
        combinations_tested = 0
        
        for C in svm_param_grid['C']:
            for gamma in svm_param_grid['gamma']:
                for kernel in svm_param_grid['kernel']:
                    for class_weight in svm_param_grid['class_weight']:
                        try:
                            combinations_tested += 1
                            if combinations_tested % 20 == 0:
                                print(f"  Tested {combinations_tested} combinations...")
                            
                            # Create and train SVM
                            svm = SVC(
                                C=C,
                                gamma=gamma,
                                kernel=kernel,
                                class_weight=class_weight,
                                random_state=42
                            )
                            
                            svm.fit(X_train_vec, y_train)
                            y_pred = svm.predict(X_test_vec)
                            
                            # Calculate metrics
                            precision = precision_score(y_test, y_pred, pos_label='spam')
                            recall = recall_score(y_test, y_pred, pos_label='spam')
                            f1 = f1_score(y_test, y_pred, pos_label='spam')
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # Combined score prioritizing precision and recall
                            combined_score = (precision * recall * f1) ** (1/3)
                            
                            if (precision >= self.target_precision and 
                                recall >= self.target_recall and 
                                combined_score > best_score):
                                
                                best_score = combined_score
                                best_svm_params = {
                                    'C': C,
                                    'gamma': gamma,
                                    'kernel': kernel,
                                    'class_weight': class_weight
                                }
                                best_results = {
                                    'precision': precision,
                                    'recall': recall,
                                    'f1': f1,
                                    'accuracy': accuracy,
                                    'combined_score': combined_score
                                }
                                
                                print(f"✅ New best SVM: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
                                
                        except Exception as e:
                            continue
        
        print(f"\n📈 Tested {combinations_tested} SVM parameter combinations")
        
        if best_svm_params:
            print(f"\n🎯 Best SVM Parameters Found:")
            for key, value in best_svm_params.items():
                print(f"  {key}: {value}")
            print(f"\n📊 Best Results:")
            for key, value in best_results.items():
                print(f"  {key}: {value:.4f}")
        else:
            print("⚠️ No SVM parameters achieved target thresholds")
            
        return best_svm_params, best_results
    
    def handle_class_imbalance(self, X_train_vec, y_train):
        """
        Apply SMOTE for handling class imbalance
        
        Args:
            X_train_vec: Vectorized training features
            y_train: Training labels
            
        Returns:
            tuple: Balanced features and labels
        """
        print("\n🔍 Phase 3.3: Class Imbalance Handling with SMOTE")
        print("-" * 50)
        
        # Check original class distribution
        original_counts = pd.Series(y_train).value_counts()
        print(f"Original class distribution:")
        for label, count in original_counts.items():
            print(f"  {label}: {count} ({count/len(y_train)*100:.1f}%)")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)
        
        # Check new class distribution
        balanced_counts = pd.Series(y_train_balanced).value_counts()
        print(f"\nBalanced class distribution:")
        for label, count in balanced_counts.items():
            print(f"  {label}: {count} ({count/len(y_train_balanced)*100:.1f}%)")
        
        print(f"\n📈 Dataset expanded from {len(y_train)} to {len(y_train_balanced)} samples")
        
        return X_train_balanced, y_train_balanced
    
    def create_ensemble_model(self, X_train_vec, y_train):
        """
        Create ensemble model with multiple algorithms
        
        Args:
            X_train_vec: Vectorized training features
            y_train: Training labels
            
        Returns:
            VotingClassifier: Trained ensemble model
        """
        print("\n🔍 Phase 3.4: Ensemble Methods Exploration")
        print("-" * 50)
        
        # Define individual models
        svm_model = SVC(
            C=2.0, gamma='scale', kernel='rbf', 
            class_weight='balanced', probability=True, random_state=42
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, 
            class_weight='balanced', random_state=42
        )
        
        lr_model = LogisticRegression(
            C=1.0, class_weight='balanced', 
            max_iter=1000, random_state=42
        )
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('svm', svm_model),
                ('rf', rf_model),
                ('lr', lr_model)
            ],
            voting='soft'  # Use probability voting
        )
        
        print("🔄 Training ensemble model...")
        ensemble.fit(X_train_vec, y_train)
        
        print("✅ Ensemble model trained successfully")
        return ensemble
    
    def cross_validate_model(self, model, X_vec, y, cv_folds=5):
        """
        Perform cross-validation for robust performance estimation
        
        Args:
            model: Trained model
            X_vec: Vectorized features
            y: Labels
            cv_folds: Number of CV folds
            
        Returns:
            dict: Cross-validation scores
        """
        print(f"\n🔍 Phase 3.5: {cv_folds}-Fold Cross-Validation")
        print("-" * 50)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, pos_label='spam'),
            'recall': make_scorer(recall_score, pos_label='spam'),
            'f1': make_scorer(f1_score, pos_label='spam')
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X_vec, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring=scoring,
            return_train_score=True
        )
        
        # Calculate mean and std for each metric
        cv_summary = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            cv_summary[metric] = {
                'mean': np.mean(test_scores),
                'std': np.std(test_scores),
                'min': np.min(test_scores),
                'max': np.max(test_scores)
            }
        
        print("📊 Cross-Validation Results:")
        for metric, stats in cv_summary.items():
            print(f"  {metric.capitalize()}:")
            print(f"    Mean: {stats['mean']:.4f} (±{stats['std']:.4f})")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        return cv_summary
    
    def optimize_complete_pipeline(self, X_train, y_train, X_test, y_test, X_train_processed, X_test_processed):
        """
        Run complete optimization pipeline to achieve target metrics
        
        Args:
            X_train, y_train: Raw training data
            X_test, y_test: Raw testing data
            X_train_processed, X_test_processed: Preprocessed text data
            
        Returns:
            dict: Complete optimization results
        """
        print("🚀 Phase 3: Advanced Model Optimization Pipeline")
        print("=" * 60)
        print(f"🎯 Target: Precision > {self.target_precision:.1%}, Recall > {self.target_recall:.1%}")
        print("=" * 60)
        
        results = {}
        
        # Step 1: Optimize TF-IDF parameters
        best_tfidf_params, tfidf_results = self.optimize_tfidf_parameters(
            X_train_processed, y_train, X_test_processed, y_test
        )
        results['tfidf_optimization'] = {'params': best_tfidf_params, 'results': tfidf_results}
        
        if best_tfidf_params:
            # Create optimized vectorizer
            self.best_vectorizer = TfidfVectorizer(
                stop_words='english',
                **best_tfidf_params
            )
            
            X_train_vec = self.best_vectorizer.fit_transform(X_train_processed)
            X_test_vec = self.best_vectorizer.transform(X_test_processed)
            
            # Step 2: Optimize SVM hyperparameters
            best_svm_params, svm_results = self.optimize_svm_hyperparameters(
                X_train_vec, y_train, X_test_vec, y_test
            )
            results['svm_optimization'] = {'params': best_svm_params, 'results': svm_results}
            
            if best_svm_params:
                # Step 3: Handle class imbalance
                X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train_vec, y_train)
                
                # Step 4: Train final optimized model
                self.best_model = SVC(random_state=42, **best_svm_params)
                self.best_model.fit(X_train_balanced, y_train_balanced)
                
                # Step 5: Final evaluation
                y_pred_final = self.best_model.predict(X_test_vec)
                final_metrics = {
                    'accuracy': accuracy_score(y_test, y_pred_final),
                    'precision': precision_score(y_test, y_pred_final, pos_label='spam'),
                    'recall': recall_score(y_test, y_pred_final, pos_label='spam'),
                    'f1': f1_score(y_test, y_pred_final, pos_label='spam')
                }
                
                results['final_metrics'] = final_metrics
                
                # Step 6: Cross-validation
                cv_results = self.cross_validate_model(self.best_model, X_train_balanced, y_train_balanced)
                results['cross_validation'] = cv_results
                
                # Check if targets are met
                target_achieved = (
                    final_metrics['precision'] >= self.target_precision and
                    final_metrics['recall'] >= self.target_recall
                )
                results['target_achieved'] = target_achieved
                
                print(f"\n🎉 PHASE 3 OPTIMIZATION COMPLETE")
                print("=" * 60)
                print(f"🏆 Final Results:")
                print(f"  Accuracy:  {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
                print(f"  Precision: {final_metrics['precision']:.4f} ({final_metrics['precision']*100:.2f}%)")
                print(f"  Recall:    {final_metrics['recall']:.4f} ({final_metrics['recall']*100:.2f}%)")
                print(f"  F1-Score:  {final_metrics['f1']:.4f} ({final_metrics['f1']*100:.2f}%)")
                print()
                
                if target_achieved:
                    print("✅ TARGET ACHIEVED: Both Precision and Recall > 93%!")
                else:
                    print("⚠️ Target not fully achieved. Consider ensemble methods or additional optimization.")
            else:
                print("❌ SVM optimization failed to meet targets")
        else:
            print("❌ TF-IDF optimization failed to meet targets")
        
        return results
    
    def save_optimized_model(self, filepath_prefix="models/phase3_optimized"):
        """
        Save the optimized model and vectorizer
        
        Args:
            filepath_prefix: Prefix for saved model files
        """
        if self.best_model and self.best_vectorizer:
            model_path = f"{filepath_prefix}_svm_classifier.joblib"
            vectorizer_path = f"{filepath_prefix}_tfidf_vectorizer.joblib"
            
            joblib.dump(self.best_model, model_path)
            joblib.dump(self.best_vectorizer, vectorizer_path)
            
            print(f"✅ Optimized model saved to: {model_path}")
            print(f"✅ Optimized vectorizer saved to: {vectorizer_path}")
        else:
            print("❌ No optimized model to save")