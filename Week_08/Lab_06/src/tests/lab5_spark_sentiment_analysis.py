'''
(i) Problem Description:
Spark-based sentiment analysis for large-scale text classification using the sentiments.csv dataset.

Input: sentiments.csv file with text and sentiment columns (-1 for negative, 1 for positive)
Output: Trained Spark ML pipeline and comprehensive evaluation metrics

(ii) Approach:
- Load data from sentiments.csv file
- Convert -1/1 labels to 0/1 format for binary classification
- Build complete ML pipeline with tokenization, stop words removal, hashing TF, IDF
- Train logistic regression model with hyperparameter tuning
- Evaluate using multiple metrics and cross-validation
'''

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def setup_spark_environment():
    '''
    Initialize and configure Spark session for sentiment analysis
    
    @return: Configured SparkSession object
    '''
    try:
        spark = SparkSession.builder \
            .appName("SparkSentimentAnalysis") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        print("✓ Spark session initialized successfully")
        return spark
    except Exception as e:
        print(f"✗ Error initializing Spark session: {str(e)}")
        return None

def load_and_preprocess_data(spark, data_path: str):
    '''
    Load sentiment data from CSV file and preprocess labels
    
    @param spark: SparkSession object
    @param data_path (str): Path to the sentiments.csv file
    @return: Preprocessed DataFrame
    '''
    try:
        print(f"Loading data from: {data_path}")
        
        # Read CSV file
        df = spark.read.csv(data_path, header=True, inferSchema=True)
        
        # Show initial schema and sample data
        print("Initial dataset schema:")
        df.printSchema()
        
        print("Sample data (first 10 rows):")
        df.show(10, truncate=False)
        
        # Check for null values
        print("Null value count per column:")
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            print(f"  {column}: {null_count}")
        
        # Drop rows with null values in critical columns
        initial_count = df.count()
        df = df.dropna(subset=["text", "sentiment"])
        after_clean_count = df.count()
        
        print(f"Data cleaning: {initial_count - after_clean_count} rows removed due to null values")
        print(f"Remaining data: {after_clean_count} rows")
        
        # Convert sentiment labels from -1/1 to 0/1
        print("Converting sentiment labels from -1/1 to 0/1...")
        df = df.withColumn("label", 
                          when(col("sentiment") == 1, 1)
                          .when(col("sentiment") == -1, 0)
                          .otherwise(None))
        
        # Remove any rows that couldn't be converted
        df = df.filter(col("label").isNotNull())
        
        # Check label distribution
        label_distribution = df.groupBy("label").count().orderBy("label")
        print("Label distribution after conversion:")
        label_distribution.show()
        
        final_count = df.count()
        print(f"Final dataset size: {final_count} rows")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return None

def build_spark_pipeline():
    '''
    Build Spark ML pipeline for text classification
    
    @return: Configured Pipeline object
    '''
    print("Building Spark ML pipeline...")
    
    # Stage 1: Tokenization - Split text into words
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    print("  ✓ Tokenizer configured")
    
    # Stage 2: Stop words removal - Remove common English stop words
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    print("  ✓ StopWordsRemover configured")
    
    # Stage 3: HashingTF - Convert words to numerical features using hashing trick
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
    print("  ✓ HashingTF configured with 1000 features")
    
    # Stage 4: IDF - Compute Inverse Document Frequency to weight important terms
    idf = IDF(inputCol="raw_features", outputCol="features")
    print("  ✓ IDF configured")
    
    # Stage 5: Logistic Regression classifier
    lr = LogisticRegression(
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.8,
        featuresCol="features",
        labelCol="label"
    )
    print("  ✓ Logistic Regression configured")
    
    # Assemble all stages into pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr])
    print("✓ Complete pipeline built with 5 stages")
    
    return pipeline

def evaluate_model(predictions, model):
    '''
    Comprehensive model evaluation using multiple metrics
    
    @param predictions: DataFrame with predictions
    @param model: Trained PipelineModel
    @return: Dictionary of evaluation metrics
    '''
    print("\n=== Model Evaluation ===")
    
    evaluators = {}
    metrics = {}
    
    # Binary classification metrics
    binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    metrics['areaUnderROC'] = binary_evaluator.evaluate(predictions)
    metrics['areaUnderPR'] = binary_evaluator.setMetricName("areaUnderPR").evaluate(predictions)
    
    # Multiclass classification metrics
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    metrics['accuracy'] = multi_evaluator.setMetricName("accuracy").evaluate(predictions)
    metrics['f1'] = multi_evaluator.setMetricName("f1").evaluate(predictions)
    metrics['weightedPrecision'] = multi_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    metrics['weightedRecall'] = multi_evaluator.setMetricName("weightedRecall").evaluate(predictions)
    
    # Display metrics
    print("Evaluation Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  F1-Score:           {metrics['f1']:.4f}")
    print(f"  Precision (Weighted): {metrics['weightedPrecision']:.4f}")
    print(f"  Recall (Weighted):    {metrics['weightedRecall']:.4f}")
    print(f"  Area Under ROC:     {metrics['areaUnderROC']:.4f}")
    print(f"  Area Under PR:      {metrics['areaUnderPR']:.4f}")
    
    # Show prediction distribution
    print("\nPrediction Distribution:")
    predictions.groupBy("prediction").count().orderBy("prediction").show()
    
    # Show some example predictions
    print("Sample Predictions (10 rows):")
    predictions.select("text", "label", "prediction", "probability").show(10, truncate=30)
    
    return metrics

def perform_cross_validation(pipeline, training_data):
    '''
    Perform cross-validation to find optimal hyperparameters
    
    @param pipeline: ML pipeline
    @param training_data: Training DataFrame
    @return: Best model from cross-validation
    '''
    print("\n=== Performing Cross-Validation ===")
    
    # Define parameter grid for tuning
    param_grid = ParamGridBuilder() \
        .addGrid(pipeline.stages[-1].regParam, [0.01, 0.1, 0.5]) \
        .addGrid(pipeline.stages[-1].elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()
    
    # Set up cross-validator
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    
    cross_val = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2
    )
    
    print("Starting 3-fold cross-validation...")
    cv_model = cross_val.fit(training_data)
    
    # Display best parameters
    best_lr = cv_model.bestModel.stages[-1]
    print("✓ Cross-validation completed")
    print(f"Best parameters:")
    print(f"  regParam: {best_lr.getRegParam()}")
    print(f"  elasticNetParam: {best_lr.getElasticNetParam()}")
    
    return cv_model.bestModel

def analyze_feature_importance(model, predictions):
    '''
    Analyze and display feature importance if available
    
    @param model: Trained PipelineModel
    @param predictions: DataFrame with predictions
    '''
    print("\n=== Feature Analysis ===")
    
    try:
        # Get the logistic regression model from pipeline
        lr_model = model.stages[-1]
        
        # Check if model has coefficients
        if hasattr(lr_model, 'coefficients'):
            coefficients = lr_model.coefficients
            print(f"Model has {len(coefficients)} features")
            print(f"Number of non-zero coefficients: {sum(1 for coef in coefficients if abs(coef) > 0.001)}")
            
            # Show extreme coefficients (most positive and most negative)
            coef_list = [(i, float(coef)) for i, coef in enumerate(coefficients)]
            coef_list.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("Top 10 most influential features (by absolute coefficient value):")
            for i, (idx, coef) in enumerate(coef_list[:10]):
                print(f"  Feature {idx}: {coef:.6f}")
                
    except Exception as e:
        print(f"Note: Feature analysis limited - {str(e)}")

def main():
    '''
    Main execution function for Spark sentiment analysis
    '''
    print("=== Spark Sentiment Analysis with sentiments.csv ===")
    
    # Setup Spark environment
    spark = setup_spark_environment()
    if spark is None:
        print("Cannot proceed without Spark session. Exiting.")
        return
    
    try:
        # Construct path to dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "dataset", "sentiments.csv")
        data_path = os.path.abspath(data_path)
        
        print(f"Looking for dataset at: {data_path}")
        
        # Check if file exists
        if not os.path.exists(data_path):
            print(f"✗ Dataset file not found: {data_path}")
            print("Please ensure the sentiments.csv file exists in the dataset/ directory")
            return
        
        # Load and preprocess data
        df = load_and_preprocess_data(spark, data_path)
        if df is None:
            print("Failed to load data. Exiting.")
            return
        
        # Split data into training and test sets (80% train, 20% test)
        print("\nSplitting data into training and test sets...")
        training_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        print(f"Training data: {training_data.count()} rows")
        print(f"Test data: {test_data.count()} rows")
        
        # Build ML pipeline
        pipeline = build_spark_pipeline()
        
        # Train model
        print("\n=== Training Model ===")
        print("Training pipeline model...")
        model = pipeline.fit(training_data)
        print("✓ Model training completed")
        
        # Make predictions
        print("\n=== Making Predictions ===")
        predictions = model.transform(test_data)
        print("✓ Predictions generated")
        
        # Evaluate model
        metrics = evaluate_model(predictions, model)
        
        # Perform cross-validation for better model (optional - can be slow)
        if training_data.count() > 100:  # Only run CV if we have enough data
            print("\n=== Advanced Model Tuning ===")
            best_model = perform_cross_validation(pipeline, training_data)
            
            # Evaluate best model
            print("\nEvaluating best model from cross-validation...")
            best_predictions = best_model.transform(test_data)
            best_metrics = evaluate_model(best_predictions, best_model)
        
        # Feature analysis
        analyze_feature_importance(model, predictions)
        
        # Pipeline information
        print("\n=== Pipeline Information ===")
        print("Pipeline stages:")
        for i, stage in enumerate(model.stages):
            print(f"  {i+1}. {stage.__class__.__name__}")
        
        print("\n✓ Spark Sentiment Analysis completed successfully!")
        
        # Save model (optional)
        # model.write().overwrite().save("./spark_sentiment_model")
        # print("Model saved to ./spark_sentiment_model")
        
    except Exception as e:
        print(f"✗ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if spark:
            spark.stop()
            print("\nSpark session stopped.")

if __name__ == "__main__":
    main()