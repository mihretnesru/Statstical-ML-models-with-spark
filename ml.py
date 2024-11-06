from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LinearSVC, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Start Spark session
spark = SparkSession.builder.appName("StudentPerformance").getOrCreate()

# Load data
df = spark.read.csv(r"/home/sat3812/Downloads/archive/StudentPerformanceFactors.csv",header=True,inferSchema=True)

# Impute missing values (Mean for numeric, Mode for categorical)
num_cols = [col for col, dtype in df.dtypes if dtype in ('int', 'double')]
cat_cols = [col for col, dtype in df.dtypes if dtype == 'string']

# Impute numeric columns with mean
for col in num_cols:
    mean_value = df.agg({col: 'mean'}).collect()[0][0]
    df = df.na.fill({col: mean_value})

# Impute categorical columns with mode
for col in cat_cols:
    mode_value = df.groupBy(col).count().orderBy('count', ascending=False).first()[0]
    df = df.na.fill({col: mode_value})

# Index categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index").fit(df) for col in cat_cols]
for indexer in indexers:
    df = indexer.transform(df)
    
# Create the 'Exam_score_classification' column based on Exam_Score
df = df.withColumn('Exam_score_classification', 
                   F.when(F.col('Exam_Score') >= 60, 'Pass').otherwise('Fail'))

# Create label column (Pass = 1, Fail = 0)
df = df.withColumn("label", (df["Exam_score_classification"] == "Pass").cast("int"))

# Apply OneHotEncoder to categorical columns
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_onehot") for col in cat_cols]
for encoder in encoders:
    df = encoder.fit(df).transform(df)

# Define features (exclude non-feature columns like Exam_Score and Exam_score_classification)
feature_columns = [col + "_onehot" for col in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Apply VectorAssembler to create the features column
df = assembler.transform(df)

# Drop the numeric "Exam_Score" column (as it's already represented in the label)
df = df.drop("Exam_Score")

# Scale features (before splitting the dataset)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Split dataset (80:20)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Define models (SVM and Random Forest)
svm = LinearSVC(featuresCol="scaled_features", labelCol="label")
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label")

# Param grid for hyperparameter tuning
paramGrid_svm = (ParamGridBuilder()
                 .addGrid(svm.regParam, [0.1, 1, 10])
                 .addGrid(svm.maxIter, [10, 50, 100])
                 .build())

paramGrid_rf = (ParamGridBuilder()
                .addGrid(rf.numTrees, [10, 50, 100])
                .addGrid(rf.maxDepth, [5, 10, 20])
                .build())

# Cross-validation
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
svm_cv = CrossValidator(estimator=svm, estimatorParamMaps=paramGrid_svm, evaluator=evaluator, numFolds=3)
rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid_rf, evaluator=evaluator, numFolds=3)

# Train models
svm_model = svm_cv.fit(train_df)
rf_model = rf_cv.fit(train_df)

# Evaluate SVM model
svm_predictions = svm_model.transform(test_df)
svm_auc = evaluator.evaluate(svm_predictions)
svm_accuracy = svm_predictions.filter(svm_predictions.label == svm_predictions.prediction).count() / float(svm_predictions.count())
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"SVM AUC: {svm_auc:.2f}")

# Evaluate Random Forest model
rf_predictions = rf_model.transform(test_df)
rf_auc = evaluator.evaluate(rf_predictions)
rf_accuracy = rf_predictions.filter(rf_predictions.label == rf_predictions.prediction).count() / float(rf_predictions.count())
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"Random Forest AUC: {rf_auc:.2f}")

# Calculate Precision, Recall, and Specificity for SVM
svm_pred = svm_predictions.select("label", "prediction").toPandas()
svm_precision = precision_score(svm_pred['label'], svm_pred['prediction'])
svm_recall = recall_score(svm_pred['label'], svm_pred['prediction'])
tn, fp, fn, tp = confusion_matrix(svm_pred['label'], svm_pred['prediction']).ravel()
svm_specificity = tn / (tn + fp)  # Specificity = TN / (TN + FP)

# Calculate Precision, Recall, and Specificity for Random Forest
rf_pred = rf_predictions.select("label", "prediction").toPandas()
rf_precision = precision_score(rf_pred['label'], rf_pred['prediction'])
rf_recall = recall_score(rf_pred['label'], rf_pred['prediction'])
tn, fp, fn, tp = confusion_matrix(rf_pred['label'], rf_pred['prediction']).ravel()
rf_specificity = tn / (tn + fp)  # Specificity = TN / (TN + FP)

# Print metrics for SVM
print(f"SVM - Precision: {svm_precision:.4f}, Recall: {svm_recall:.4f}, Specificity: {svm_specificity:.4f}, AUC: {svm_auc:.4f}")

# Print metrics for Random Forest
print(f"Random Forest - Precision: {rf_precision:.4f}, Recall: {rf_recall:.4f}, Specificity: {rf_specificity:.4f}, AUC: {rf_auc:.4f}")

# Plot ROC curves
plt.figure(figsize=(10, 6))

for model_name, predictions, auc_value in zip(["SVM", "Random Forest"], 
                                              [svm_predictions, rf_predictions], 
                                              [svm_auc, rf_auc]):
    y_true = predictions.select("label").toPandas()
    y_scores = predictions.select("prediction").toPandas()
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_value:.2f})')

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Stop Spark session
spark.stop()
