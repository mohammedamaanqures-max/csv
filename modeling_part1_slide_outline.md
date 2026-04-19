# PowerPoint Slide Outline (Modeling Part 1)

1. **Title Slide**
   - Project: Airline Passenger Satisfaction Modeling Part 1
   - Focus: Class Imbalance + Feature Selection

2. **Data Preparation Summary (Completed Previously)**
   - Cleaning, encoding, transformation recap
   - Final modeling target: satisfaction

3. **Train/Test Split**
   - 80/20 stratified split
   - Reproducibility via random_state
   - Show train/test class counts

4. **Class Imbalance: Baseline**
   - Pre-balancing class distribution chart
   - Why imbalance matters for classification

5. **Imbalance Technique 1: SMOTE**
   - Concept: synthetic minority generation
   - Before/after class distribution
   - Strengths and trade-offs

6. **Imbalance Technique 2: Random Undersampling**
   - Concept: reduce majority samples
   - Before/after class distribution
   - Strengths and trade-offs

7. **Feature Selection Method 1: LASSO**
   - L1 regularization concept
   - Top coefficient bar chart
   - Key selected features

8. **Feature Selection Method 2: Random Forest Importance**
   - Tree-based importance concept
   - Top feature importance bar chart
   - Key selected features

9. **Results Table (Required Format)**
   - Feature Selection Method
   - Balancing Technique
   - Important Features

10. **Interpretation & Key Findings**
    - How balancing changed training data
    - Consistent important features across methods
    - Differences between LASSO and RF rankings

11. **Next Steps**
    - Modeling Part 2: train/evaluate classifiers
    - Compare metrics (precision, recall, F1, ROC-AUC)
