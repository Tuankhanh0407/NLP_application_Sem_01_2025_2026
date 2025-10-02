package com.harito.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer, Normalizer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.sql.Row
import java.io.{File, PrintWriter}

object Lab17_NLPPipeline {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)

    // 1. --- Customizable Document Limit ---
    val limitDocuments = 1000 // Easily change this value to process different numbers of documents
    println(s"Processing limit set to: $limitDocuments documents")

    // --- Performance Measurement Variables ---
    var readStartTime, readEndTime: Long = 0
    var fitStartTime, fitEndTime: Long = 0
    var transformStartTime, transformEndTime: Long = 0
    var similarityStartTime, similarityEndTime: Long = 0
    var writeStartTime, writeEndTime: Long = 0

    // 2. --- Read Dataset with Performance Measurement ---
    readStartTime = System.nanoTime()
    val dataPath = "../data/c4-train.00000-of-01024-30K.json.gz"
    val initialDF = spark.read.json(dataPath).limit(limitDocuments)
    val recordCount = initialDF.count()
    readEndTime = System.nanoTime()
    val readDuration = (readEndTime - readStartTime) / 1e9d
    
    println(s"Successfully read $recordCount records in ${readDuration} seconds.")
    initialDF.printSchema()
    println("\nSample of initial DataFrame:")
    initialDF.show(5, truncate = false)

    // --- Pipeline Stages Definition ---

    // Tokenization
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']")

    // Stop Words Removal
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    // Vectorization (Term Frequency)
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(20000)

    // Vectorization (Inverse Document Frequency)
    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("tfidf_features")

    // 3. --- Vector Normalization ---
    val normalizer = new Normalizer()
      .setInputCol(idf.getOutputCol)
      .setOutputCol("features")
      .setP(2.0) // L2 normalization for cosine similarity compatibility

    // Assemble the Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer))

    // --- Time the main operations ---

    println("\nFitting the NLP pipeline...")
    fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(initialDF)
    fitEndTime = System.nanoTime()
    val fitDuration = (fitEndTime - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")

    println("\nTransforming data with the fitted pipeline...")
    transformStartTime = System.nanoTime()
    val transformedDF = pipelineModel.transform(initialDF)
    transformedDF.cache()
    val transformCount = transformedDF.count()
    transformEndTime = System.nanoTime()
    val transformDuration = (transformEndTime - transformStartTime) / 1e9d
    println(f"--> Data transformation of $transformCount records took $transformDuration%.2f seconds.")

    // Calculate actual vocabulary size after tokenization and stop word removal
    val actualVocabSize = transformedDF
      .select(explode($"filtered_tokens").as("word"))
      .filter(length($"word") > 1)
      .distinct()
      .count()
    println(s"--> Actual vocabulary size after tokenization and stop word removal: $actualVocabSize unique terms.")

    // --- Show and Save Results ---
    println("\nSample of transformed data:")
    transformedDF.select("text", "tokens", "filtered_tokens", "features").show(5, truncate = 50)

    // 4. --- Find Similar Documents using Cosine Similarity ---
    println("\nFinding similar documents...")
    similarityStartTime = System.nanoTime()
    
    // Select a query document (using the first document as example)
    val queryDoc = transformedDF.first()
    val queryFeatures = queryDoc.getAs[Vector]("features")
    val queryText = queryDoc.getAs[String]("text")
    
    println(s"Query Document (first document in dataset):")
    println(s"Text: ${queryText.substring(0, Math.min(queryText.length, 150))}...")
    
    // Calculate cosine similarity between query document and all others
    val similarityDF = transformedDF.rdd.map { row =>
      val docId = row.fieldIndex("text") // Using text as identifier
      val docFeatures = row.getAs[Vector]("features")
      val docText = row.getAs[String]("text")
      
      // Calculate cosine similarity: dot product of normalized vectors
      val similarity = if (queryFeatures.numNonzeros > 0 && docFeatures.numNonzeros > 0) {
        val dotProduct = queryFeatures.toArray.zip(docFeatures.toArray).map { 
          case (a, b) => a * b 
        }.sum
        dotProduct // Since vectors are L2 normalized, dot product = cosine similarity
      } else {
        0.0
      }
      
      (docText, similarity, row)
    }.toDF("document_text", "cosine_similarity", "original_row")
    .filter($"cosine_similarity" < 0.999) // Exclude the query document itself
    .orderBy(desc("cosine_similarity"))
    .limit(5)
    
    println("\nTop 5 most similar documents:")
    similarityDF.select("document_text", "cosine_similarity").show(5, truncate = 100)
    
    similarityEndTime = System.nanoTime()
    val similarityDuration = (similarityEndTime - similarityStartTime) / 1e9d
    println(f"--> Similarity computation took $similarityDuration%.2f seconds.")

    val n_results = 20
    val results = transformedDF.select("text", "tokens", "filtered_tokens", "features").take(n_results)

    // 5. --- Write Metrics and Results to Separate Files ---
    writeStartTime = System.nanoTime()

    // Write metrics to the log folder
    val log_path = "../log/lab17_metrics.log"
    new File(log_path).getParentFile.mkdirs()
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("--- Performance Metrics ---")
      logWriter.println(f"Document processing limit: $limitDocuments")
      logWriter.println(f"Data reading duration: $readDuration%.2f seconds")
      logWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      logWriter.println(f"Data transformation duration: $transformDuration%.2f seconds")
      logWriter.println(f"Similarity computation duration: $similarityDuration%.2f seconds")
      logWriter.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize unique terms")
      logWriter.println(s"HashingTF numFeatures set to: 20000")
      if (20000 < actualVocabSize) {
        logWriter.println(s"Note: numFeatures (20000) is smaller than actual vocabulary size ($actualVocabSize). Hash collisions are expected.")
      }
      logWriter.println(s"Normalization: L2 normalization applied for cosine similarity")
      logWriter.println(s"Metrics file generated at: ${new File(log_path).getAbsolutePath}")
      logWriter.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
      
      // Add similarity results to metrics
      logWriter.println("\n--- Similarity Analysis ---")
      logWriter.println(s"Query document: ${queryText.substring(0, Math.min(queryText.length, 200))}...")
      val topSimilarDocs = similarityDF.select("document_text", "cosine_similarity").collect()
      logWriter.println("Top 5 similar documents:")
      topSimilarDocs.foreach { row =>
        val docText = row.getAs[String]("document_text")
        val similarity = row.getAs[Double]("cosine_similarity")
        logWriter.println(f"Similarity: $similarity%.4f - Text: ${docText.substring(0, Math.min(docText.length, 150))}...")
      }
      
      println(s"\nSuccessfully wrote metrics to $log_path")
    } finally {
      logWriter.close()
    }

    // Write data results to the results folder
    val result_path = "../results/lab17_pipeline_output.txt"
    new File(result_path).getParentFile.mkdirs()
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println(s"--- NLP Pipeline Output (First $n_results results) ---")
      resultWriter.println(s"Document processing limit: $limitDocuments")
      resultWriter.println(s"Output file generated at: ${new File(result_path).getAbsolutePath}\n")
      
      results.foreach { row =>
        val text = row.getAs[String]("text")
        val tokens = row.getAs[Seq[String]]("tokens")
        val filteredTokens = row.getAs[Seq[String]]("filtered_tokens")
        val features = row.getAs[Vector]("features")
        
        resultWriter.println("="*80)
        resultWriter.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"Tokens: ${tokens.take(20).mkString(", ")}${if (tokens.length > 20) "..." else ""}")
        resultWriter.println(s"Filtered Tokens: ${filteredTokens.take(20).mkString(", ")}${if (filteredTokens.length > 20) "..." else ""}")
        resultWriter.println(s"Normalized TF-IDF Vector: ${features.toString}")
        resultWriter.println("="*80)
        resultWriter.println()
      }
      
      // Add similarity results
      resultWriter.println("\n" + "="*80)
      resultWriter.println("SIMILAR DOCUMENTS ANALYSIS")
      resultWriter.println("="*80)
      resultWriter.println(s"Query Document: ${queryText.substring(0, Math.min(queryText.length, 200))}...")
      resultWriter.println("\nTop 5 Most Similar Documents:")
      
      val topSimilarDocs = similarityDF.select("document_text", "cosine_similarity").collect()
      topSimilarDocs.zipWithIndex.foreach { case (row, index) =>
        val docText = row.getAs[String]("document_text")
        val similarity = row.getAs[Double]("cosine_similarity")
        resultWriter.println(s"\n${index + 1}. Cosine Similarity: $similarity")
        resultWriter.println(s"Document: ${docText.substring(0, Math.min(docText.length, 200))}...")
      }
      
      println(s"Successfully wrote $n_results results and similarity analysis to $result_path")
    } finally {
      resultWriter.close()
    }
    
    writeEndTime = System.nanoTime()
    val writeDuration = (writeEndTime - writeStartTime) / 1e9d
    println(f"--> Results writing took $writeDuration%.2f seconds.")

    // Print summary of all performance metrics
    println("\n" + "="*50)
    println("PERFORMANCE SUMMARY")
    println("="*50)
    println(f"Data Reading:        $readDuration%8.2f seconds")
    println(f"Pipeline Fitting:    $fitDuration%8.2f seconds")
    println(f"Data Transformation: $transformDuration%8.2f seconds")
    println(f"Similarity Compute:  $similarityDuration%8.2f seconds")
    println(f"Results Writing:     $writeDuration%8.2f seconds")
    println("-"*50)
    val totalTime = readDuration + fitDuration + transformDuration + similarityDuration + writeDuration
    println(f"TOTAL TIME:          $totalTime%8.2f seconds")
    println("="*50)

    spark.stop()
    println("Spark Session stopped.")
  }
}