'''
(i) Problem Description:
This program implements a Spark NLP pipeline for text processing and feature extraction from the C4 dataset.
The pipeline performs tokenization, stop words removal, and TF-IDF vectorization on text data.

Input: 
- Compressed JSON file (c4-train.00000-of-01024-30K.json.gz) containing text documents
- Configuration parameters for the NLP pipeline

Output:
- Feature vectors representing the processed text data
- Log files recording the pipeline execution process
- Saved results in the specified output directory

(ii) Approach:
The solution uses PySpark MLlib to build a multi-stage NLP pipeline:
1. Data loading and preprocessing using Spark DataFrames
2. Text tokenization using RegexTokenizer
3. Stop words removal using StopWordsRemover
4. Feature vectorization using HashingTF and IDF
5. Result persistence and logging
The implementation follows modular design principles for maintainability and scalability.
'''

import os
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.types import StructType, StructField, StringType


class SparkSessionManager:
    '''
    Manages Spark session creation and configuration
    @param app_name (str): Name of the Spark application
    @param master (str): Spark master URL (default: "local[*]")
    @return spark_session (SparkSession): Configured Spark session object
    '''
    
    @staticmethod
    def create_spark_session(app_name="SparkNLPipeline", master="local[*]"):
        '''
        Creates and configures a Spark session
        @param app_name (str): Name of the Spark application
        @param master (str): Spark master URL
        @return spark_session (SparkSession): Configured Spark session
        '''
        try:
            spark = SparkSession.builder \
                .appName(app_name) \
                .master(master) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            spark.sparkContext.setLogLevel("WARN")
            return spark
        except Exception as e:
            raise RuntimeError(f"Failed to create Spark session: {str(e)}")


class DataLoader:
    '''
    Handles data loading operations from various sources
    '''
    
    @staticmethod
    def load_json_data(spark_session, file_path, limit=None):
        '''
        Loads JSON data into Spark DataFrame
        @param spark_session (SparkSession): Active Spark session
        @param file_path (str): Path to the JSON file
        @param limit (int): Optional limit for number of records to load
        @return dataframe (DataFrame): Spark DataFrame containing loaded data
        '''
        try:
            logger.info(f"Loading data from: {file_path}")
            
            # Define schema for C4 dataset
            schema = StructType([
                StructField("text", StringType(), True),
                StructField("url", StringType(), True),
                StructField("timestamp", StringType(), True)
            ])
            
            dataframe = spark_session.read.json(file_path, schema=schema)
            
            # Select only text column for NLP processing
            if "text" in dataframe.columns:
                dataframe = dataframe.select("text")
            
            if limit:
                dataframe = dataframe.limit(limit)
                
            record_count = dataframe.count()
            logger.info(f"Successfully loaded {record_count} records")
            
            return dataframe
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


class NLPPipelineBuilder:
    '''
    Builds and configures the NLP pipeline stages
    '''
    
    @staticmethod
    def create_tokenizer_stage(input_col="text", output_col="words"):
        '''
        Creates tokenizer stage for text processing
        @param input_col (str): Input column name
        @param output_col (str): Output column name
        @return tokenizer (RegexTokenizer): Configured tokenizer object
        '''
        return RegexTokenizer(
            inputCol=input_col,
            outputCol=output_col,
            pattern="\\W+",  # Match non-word characters
            gaps=True,
            minTokenLength=2  # Minimum token length
        )
    
    @staticmethod
    def create_stopwords_remover_stage(input_col="words", output_col="filtered_words"):
        '''
        Creates stop words remover stage
        @param input_col (str): Input column name
        @param output_col (str): Output column name
        @return stopwords_remover (StopWordsRemover): Configured stop words remover
        '''
        return StopWordsRemover(
            inputCol=input_col,
            outputCol=output_col
        )
    
    @staticmethod
    def create_tfidf_stages(input_col="filtered_words", num_features=20000):
        '''
        Creates TF-IDF vectorization stages
        @param input_col (str): Input column name
        @param num_features (int): Number of features for HashingTF
        @return stages (list): List containing HashingTF and IDF stages
        '''
        hashing_tf = HashingTF(
            inputCol=input_col,
            outputCol="raw_features",
            numFeatures=num_features
        )
        
        idf = IDF(
            inputCol="raw_features",
            outputCol="features"
        )
        
        return [hashing_tf, idf]
    
    @staticmethod
    def build_pipeline():
        '''
        Builds complete NLP pipeline with all stages
        @return pipeline (Pipeline): Configured ML pipeline
        '''
        tokenizer = NLPPipelineBuilder.create_tokenizer_stage()
        stopwords_remover = NLPPipelineBuilder.create_stopwords_remover_stage()
        tfidf_stages = NLPPipelineBuilder.create_tfidf_stages()
        
        all_stages = [tokenizer, stopwords_remover] + tfidf_stages
        
        return Pipeline(stages=all_stages)


class ResultSaver:
    '''
    Handles saving of pipeline results and output
    '''
    
    @staticmethod
    def save_results(dataframe, output_path):
        '''
        Saves processed results to output file
        @param dataframe (DataFrame): DataFrame containing results
        @param output_path (str): Path for output directory
        @return success (bool): True if save operation successful
        '''
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logger.info(f"Saving results to: {output_path}")
            
            # Select relevant columns for output
            output_cols = ["text", "words", "filtered_words", "features"]
            available_cols = [col for col in output_cols if col in dataframe.columns]
            
            result_df = dataframe.select(available_cols)
            
            # Write results
            result_df.write \
                .mode("overwrite") \
                .option("compression", "gzip") \
                .json(output_path)
            
            logger.info("Results saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False


class LoggerSetup:
    '''
    Configures logging for the application
    '''
    
    @staticmethod
    def setup_logging(log_directory="logs"):
        '''
        Sets up logging configuration
        @param log_directory (str): Directory for log files
        @return logger (Logger): Configured logger object
        '''
        try:
            os.makedirs(log_directory, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_directory, f"spark_nlp_pipeline_{timestamp}.log")
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            
            return logging.getLogger(__name__)
            
        except Exception as e:
            print(f"Failed to setup logging: {str(e)}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)


class PipelineExecutor:
    '''
    Executes the complete NLP pipeline workflow
    '''
    
    @staticmethod
    def execute_pipeline():
        '''
        Main pipeline execution method coordinating all components
        @return success (bool): True if pipeline execution successful
        '''
        spark = None
        try:
            logger.info("Starting Spark NLP Pipeline Execution")
            start_time = datetime.now()
            
            # Initialize Spark session
            spark = SparkSessionManager.create_spark_session()
            logger.info("Spark session created successfully")
            
            # Define file paths
            input_file = "c4-train.00000-of-01024-30K.json.gz"
            output_path = "results/pipeline_output"
            
            # Load data
            data = DataLoader.load_json_data(spark, input_file, limit=1000)
            
            # Build pipeline
            pipeline = NLPPipelineBuilder.build_pipeline()
            logger.info("NLP pipeline built with all stages")
            
            # Fit and transform data
            logger.info("Fitting pipeline to data...")
            pipeline_model = pipeline.fit(data)
            
            logger.info("Transforming data...")
            transformed_data = pipeline_model.transform(data)
            
            # Show sample results
            logger.info("Sample of transformed data:")
            transformed_data.select("text", "words", "filtered_words").show(5, truncate=True)
            transformed_data.select("features").show(5, truncate=False)
            
            # Save results
            save_success = ResultSaver.save_results(transformed_data, output_path)
            
            # Log execution summary
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            logger.info(f"Pipeline execution completed in {execution_time}")
            logger.info(f"Total records processed: {transformed_data.count()}")
            
            if save_success:
                logger.info("Pipeline execution completed successfully")
                return True
            else:
                logger.error("Pipeline execution completed with errors in saving")
                return False
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return False
            
        finally:
            if spark:
                spark.stop()
                logger.info("Spark session stopped")


def main():
    '''
    Main method to execute the Spark NLP pipeline
    Coordinates the entire workflow and handles execution
    @return None
    '''
    try:
        success = PipelineExecutor.execute_pipeline()
        
        if success:
            print("Spark NLP Pipeline executed successfully!")
            logger.info("Application completed successfully")
        else:
            print("Spark NLP Pipeline execution completed with errors")
            logger.error("Application completed with errors")
            
    except Exception as e:
        error_msg = f"Application failed: {str(e)}"
        print(error_msg)
        if 'logger' in globals():
            logger.error(error_msg)


# Global logger initialization
logger = LoggerSetup.setup_logging()

if __name__ == "__main__":
    '''
    Entry point of the Spark NLP Pipeline application
    '''
    main()