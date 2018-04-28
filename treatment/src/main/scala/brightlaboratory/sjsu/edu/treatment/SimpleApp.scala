/* SimpleApp.scala */

package brightlaboratory.sjsu.edu.treatment;

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions.typedLit


object PredictApp {
  /*
    def main(args:Array[String]){
      val conf = new SparkConf()
      conf.set("spark.master", "local")
      conf.set("spark.app.name", "exampleSpark")
      val sc = new SparkContext(conf)
      val sparkSession = SparkSession.builder.
        master("local")
        .appName("spark session example")
        .getOrCreate()
      SimpleApp.createDataframe(sparkSession)
    }
      */
}

object SimpleApp {

  def createDataframe(spark: SparkSession) = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val df = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("data.csv").getPath)

    // Convert all columns into integer
    val someCastedDF = (df.columns.toBuffer).foldLeft(df)((current, c) =>current.withColumn(c, col(c).cast("int")))

    //Select 1) Age 2) Gender 3) Race 4) Ethnic 5) Marital Status 6) Education 7) Employment status 8) STFIPS
    val df_new = df.select("AGE", "GENDER","RACE","ETHNIC","MARSTAT","EDUC","STFIPS","SUB1")
    //someCastedDF.printSchema()

    //preOrigDf.show(numRows = 1)
    someCastedDF.createOrReplaceTempView("INPUT")

    // get count of total success and failures
    val total_success = spark.sql("select * from INPUT where REASON = 1").count()
    val total_failure = spark.sql("select count(*) as TotalFailure, REASON from INPUT where REASON <> 1 group by REASON").agg(sum("TotalFailure" )).head()

    //get success and failure for each SERVSETD categories
    val SERVSETD_success = spark.sql("select SERVSETD, count(*) as count from INPUT WHERE REASON = 1 GROUP BY SERVSETD ORDER BY count DESC")
    val SERVSETD_failure = spark.sql("select SERVSETD, count(*) as count from INPUT WHERE REASON <> 1 GROUP BY SERVSETD ORDER BY count DESC")

    //for each SERVSETD category calculate it's success and failure probability
    SERVSETD_success.createOrReplaceTempView("SERVSETD_success")
    SERVSETD_failure.createOrReplaceTempView("SERVSETD_failure")
    val success_prob = spark.sql(s"select SERVSETD, Round(count/${total_success}, 3) as s_prob from SERVSETD_success")
    val failure_prob = spark.sql(s"select SERVSETD, Round(count/${total_failure{0}}, 3) as f_prob from SERVSETD_failure")
    //success_prob.show()
    //failure_prob.show()

    //calculate rank of each SERVSETD
    success_prob.createOrReplaceTempView("success_prob")
    failure_prob.createOrReplaceTempView("failure_prob")
    val SERVSETD_rankDf = spark.sql("select a.SERVSETD, Round(a.s_prob/b.f_prob , 3) " +
      "as score from success_prob a inner join failure_prob b ON a.SERVSETD = b.SERVSETD order by score ASC")
    SERVSETD_rankDf.createOrReplaceTempView("score_table")
    //val finalDF = SERVSETD_rankDf.withColumn("rowd",monotonically_increasing_id()+1)
    val finalDF = spark.sql("select SERVSETD, score, RANK() OVER (ORDER BY score ASC) as rank from score_table")
    //finalDF.show()
    //println("Rank of SERVSETD")

    //    predictTreatmentSuccess(someCastedDF, NEW_SERVSETD_FAILURE)

    predictTreatmentSuccessAndModifySERVSETD(someCastedDF, finalDF)

    //predictTreatmentCompletion(someCastedDF)
    //multilayerPerceptronClassifier(df_new)
    //kMeansClustering(someCastedDF)
    // calculateColumnValuePercentage(someCastedDF)

  }



  //Calculate percentage of each unique range in a column

  def calculateColumnValuePercentage(dataFrame: DataFrame)={

    //Calculating total records in a df
    val total_records= dataFrame.count()


    // Count of unique values in columns and Adding Actual Name Labels to column AGE,GENDER,RACE. Creating three different dataframes

    //AGE
    val countAge=dataFrame.groupBy("AGE").count()
    val ageDF=assignOriginalColumnLabelsAGE(countAge)

    //GENDER
    val countGENDER=dataFrame.groupBy("GENDER").count()
    val genderDF=assignOriginalColumnLabelsGENDER(countGENDER)

    //RACE
    val countRace=dataFrame.groupBy("RACE").count()
    val raceDF=assignOriginalColumnLabelsRACE(countRace)


    // Calculating percentage of each unique value of AGE
    def func_percentAge = udf((age:Int, count: Int, totalCount: Long) => {
      val totalCountDouble = totalCount.toDouble
      if (age == 2)      (count/totalCountDouble)*100
      else if (age == 3) (count/totalCountDouble)*100
      else if (age == 4) (count/totalCountDouble)*100
      else if (age == 5) (count/totalCountDouble)*100
      else if (age == 6) (count/totalCountDouble)*100
      else if (age == 7) (count/totalCountDouble)*100
      else if (age == 8) (count/totalCountDouble)*100
      else if (age == 9) (count/totalCountDouble)*100
      else if (age == 10) (count/totalCountDouble)*100
      else if (age == 11) (count/totalCountDouble)*100
      else if (age == 12) (count/totalCountDouble)*100
      else 0
    })


    // val new_df1 = ageDF.withColumn("percentage", func_percentAge(col("AGE"),col("count")))
    val new_df1 = ageDF.withColumn("percentage", func_percentAge(col("AGE"),col("count"), lit(total_records)))

    //Writing df into a file
    new_df1.coalesce(1).write.option("header", "true").csv("/Users/anujatike/Documents/sem3/RA/Project2/HistogramData/ageData.csv")
  }


  def assignOriginalColumnLabelsAGE(dataFrame: DataFrame)= {

    // Adding Actual Name Labels to column AGE

    val func_age: (Int => String) = (age: Int) => {
      if      (age == 2) "12-14"
      else if (age == 3) "15-17"
      else if (age == 4) "18-20"
      else if (age == 5) "21-24"
      else if (age == 6) "25-29"
      else if (age == 7) "30-34"
      else if (age == 8) "35-39"
      else if (age == 9) "40-44"
      else if (age == 10) "45-49"
      else if (age == 11) "50-54"
      else if (age == 12) "55 AND OVER"
      else "MISSING/UNKNOWN/NOT COLLECTED/INVALID"
    }
    val func1 = udf(func_age)

    val new_df1 = dataFrame.withColumn("originalAGELabels", func1(col("AGE")))


    new_df1
  }

  def assignOriginalColumnLabelsGENDER(dataFrame: DataFrame)= {
    // Adding Actual Name Labels to column GENDER

    val func_gender: (Int => String) = (gender: Int) => {
      if (gender == 1) "MALE"
      else if (gender == 2) "FEMALE"
      else "MISSING/UNKNOWN/NOT COLLECTED/INVALID"
    }
    val func2 = udf(func_gender)
    val new_df2 = dataFrame.withColumn("originalGENDERLabels", func2(col("GENDER")))

    new_df2
  }

  def assignOriginalColumnLabelsRACE(dataFrame: DataFrame)= {
    //Adding Actual Name Labels to column RACE

    val func_race: (Int => String) = (race: Int) => {
      if  (race == 1) "ALASKA NATIVE (ALEUT, ESKIMO, INDIAN)"
      else if (race==2) "AMERICAN INDIAN (OTHER THAN ALASKA NATIVE)"
      else if (race==3) "ASIAN OR PACIFIC ISLANDER"
      else if (race==4) "BLACK OR AFRICAN AMERICAN"
      else if (race==5) "WHITE"
      else if (race==13) "ASIAN"
      else if (race==20) "OTHER SINGLE RACE"
      else if (race==21) "TWO OR MORE RACES"
      else if (race==23) "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER"
      else "MISSING/UNKNOWN/NOT COLLECTED/INVALID"
    }
    val func3 = udf(func_race)
    val new_df3=dataFrame.withColumn("originalRACELabels", func3(col("RACE")))

    new_df3
  }



  def kMeansClustering(dataFrame: DataFrame)= {


    //creating features column
    val assembler = new VectorAssembler()
      .setInputCols(Array("AGE", "GENDER", "STFIPS", "SERVSETD", "NOPRIOR", "SUB1", "SUB2", "SUB3"))
      .setOutputCol("features")


    val output = assembler.transform(dataFrame)



    val splitSeed = 5043
    val Array(trainingData, testData) = output.randomSplit(Array(0.7, 0.3), splitSeed)

    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(trainingData)

    println("kmeans model: " + model)
    println("kmeans model.summary.clusterSizes: " + model.summary.clusterSizes.deep)

    val df4 = model.transform(testData)

    df4.show(10, truncate = false)
    df4.printSchema()

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(trainingData)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    println("Cluster centers:")
    model.clusterCenters.foreach(println)



  }

  def predictTreatmentSuccessAndModifySERVSETD(preOrigDf: DataFrame, rankDf:
  DataFrame) = {
    // If the treatment is completed, it is 1, everything else is set to 0
    val column = "REASON"
    val origDf = preOrigDf.withColumn(column, when(col(column).notEqual(1), 0).otherwise(1))
    val changeDf = origDf.select(origDf("CASEID"),origDf("SERVSETD") as "cSERVSETD", origDf("METHUSE"), origDf("LOS")  as "cLOS", origDf("SUB1"),  origDf("ROUTE1"), origDf("NUMSUBS"), origDf("DSMCRIT"), origDf("REASON"))

    var labelIndexer = new StringIndexer().setInputCol("REASON").setOutputCol("label")
    var labelIndexerModel = labelIndexer.fit(changeDf)
    var df = labelIndexerModel.transform(changeDf)
    //df.show(10)

    var assembler = new VectorAssembler().setInputCols(Array("CASEID", "cSERVSETD", "METHUSE", "cLOS", "SUB1",
      "ROUTE1", "NUMSUBS", "DSMCRIT")).setOutputCol("features")
    var df2 = assembler.transform(df)

    // Split the data into train and test
    val splits = df2.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val trainingData = splits(0)
    val testData = splits(1)


    val classifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(8)
      .setNumTrees(20)
      .setMaxBins(100)
      .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

    println("For the original dataset: ")
    var model = classifier.fit(trainingData)
//    println("Random Forest Regresser model: " + model.toDebugString)
    println("model.featureImportances: " + model.featureImportances)

    var predictions = model.transform(testData)
    println("PREDICTIONS:")
    predictions.show(10)
    println("testData.count(): " + testData.count())

    var converter = new IndexToString().setInputCol("prediction")
      .setOutputCol("originalValue")
      .setLabels(labelIndexerModel.labels)
    var df3 = converter.transform(predictions)

    df3.createOrReplaceTempView("Table1")

    var predictionAndLabels = predictions.select("prediction", "label")
    var evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))

    // Consider all rows whose PREDICTION = 0
    // Consider all rows whose PREDICTION = 0
    var failedRows = predictions.where(predictions("prediction") === 0.0)
    //failedRows.show(10)
    var iter_test = increaseSERVSETD(failedRows, model, rankDf)
    iter_test.show(5)
    var i = 2
    //while(iter_test.count() != 0){
    while(i < 5){
      println("    ROUND :  " + i)
      i = i + 1
      iter_test = iter_test.drop("cSERVSETD","prediction","rawPrediction", "probability","features")
      iter_test = iter_test.withColumnRenamed("cSERVSETDNew", "cSERVSETD")

      assembler = new VectorAssembler().setInputCols(Array("CASEID", "cSERVSETD", "METHUSE", "cLOS", "SUB1",
        "ROUTE1", "NUMSUBS", "DSMCRIT")).setOutputCol("features")
      iter_test = assembler.transform(iter_test)
      iter_test.show(10)

      model = classifier.fit(iter_test)
      //println("Random Forest Regresser model: " + model.toDebugString)
      println("model.featureImportances: " + model.featureImportances)

      predictions = model.transform(iter_test)
      println("PREDICTIONS:")
      predictions.show(10)
      println("iter_test.count(): " + iter_test.count())
      df3 = converter.transform(predictions)

      df3.createOrReplaceTempView("Table1")

      predictionAndLabels = predictions.select("prediction", "label")
      evaluator = new MulticlassClassificationEvaluator()
        .setMetricName("accuracy")
      iter_test.createOrReplaceTempView("iterative_test_table")

      println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
      println("prediction 1's count" + predictions.where(predictions("prediction")=== 1.0).count())
      var successfulConversion =  predictions.where(predictions("prediction")=== 1.0).count()/iter_test.count()
      println("Successful Conversion Percent" + successfulConversion)
      failedRows = predictions.where(predictions("prediction") === 0.0)
      iter_test = increaseSERVSETD(failedRows, model, rankDf)

    }

  }


  def increaseServiceSetting = udf((cSERVSETD:Int, SERVSETD_to_rank_map:
  Seq[Int]) => {

//    SERVSETD_to_rank_map.zipWithIndex.foreach(v => println("ZIP: " + v))
//    sys.exit(0)

//    val nextIndex = SERVSETD_to_rank_map.zipWithIndex.filter(v => v._1 ==
//      cSERVSETD)
//      .head._2 + 1
    val nextIndex = SERVSETD_to_rank_map.indexOf(cSERVSETD) + 1
    //println("nextIndex: " + nextIndex)
    SERVSETD_to_rank_map{nextIndex}
  })

  def increaseSERVSETD(failedRows: DataFrame, model:
  RandomForestClassificationModel, rankDf: DataFrame): DataFrame = {

    val maxRowd = rankDf.select(max("rank")).collect() {
      0
    }.getInt(0)

    val hightestSERSETD = rankDf.select("SERVSETD").where(rankDf("rank") ===
      maxRowd)
      .collect() {
        0
      }
      .getInt(0)

    // We will filter highest ranked rows because their service setting
    // cannot be increased
    val promisingRows = failedRows.filter(failedRows("cSERVSETD") =!=
      hightestSERSETD)

    //rankDf.printSchema()
    //println("rankDf:")
    //rankDf.show(100, truncate = false)
    val SERVSETD_to_rank_map =
      rankDf.drop("rank").collect().map(row => row.getInt(0))

    SERVSETD_to_rank_map.take(10).foreach(v => println("ROW: " + v))
//    sys.exit(0)


//    promisingRows.drop("cSERVSETD")

   val SERVSETD_to_rank_list = SERVSETD_to_rank_map.toSeq
    print("The list: ",SERVSETD_to_rank_list)

   val promisingRowsMod = promisingRows.withColumn("cSERVSETDNew",
      increaseServiceSetting
    (promisingRows("cSERVSETD"), typedLit(SERVSETD_to_rank_list) ))

    promisingRowsMod.show(5, truncate = false)
    return(promisingRowsMod)

    // Run the rows of promisingRowsMod through the model to see how many of
    // those are not SUCCESS

  }



  def predictTreatmentSuccess(preOrigDf: DataFrame, NEW_SERVSETD_FAILURE: DataFrame) = {

    // If the treatment is completed, it is 1, everything else is set to 0
    val column = "REASON"
    val origDf = preOrigDf.withColumn(column, when(col(column).notEqual(1), 0).otherwise(1))
    val changeDf = origDf.select(origDf("CASEID"),origDf("SERVSETD") as "cSERVSETD", origDf("METHUSE"), origDf("LOS")  as "cLOS", origDf("SUB1"),  origDf("ROUTE1"), origDf("NUMSUBS"), origDf("DSMCRIT"), origDf("REASON"))

    val labelIndexer = new StringIndexer().setInputCol("REASON").setOutputCol("label")
    val labelIndexerModel = labelIndexer.fit(changeDf)
    val df = labelIndexerModel.transform(changeDf)
    //df.show(10)

    val assembler = new VectorAssembler().setInputCols(Array("CASEID", "cSERVSETD", "METHUSE", "cLOS", "SUB1",
      "ROUTE1", "NUMSUBS", "DSMCRIT")).setOutputCol("features")
    val trainingData = assembler.transform(df)

    //creating the test data using NEW_SERVSETD_FAILURE
    val test_column = "REASON"
    // If the treatment is completed, it is 1, everything else is set to 0
    val newdf = NEW_SERVSETD_FAILURE.withColumn(test_column, when(col(test_column).notEqual(1), 0).otherwise(1))
    val changetestDf = newdf.select(newdf("CASEID"), newdf("DERIVED_SERV")  as "cSERVSETD", newdf("METHUSE"), newdf("LOS")  as "cLOS", newdf("SUB1"),  newdf("ROUTE1"), newdf("NUMSUBS"), newdf("DSMCRIT"), newdf("REASON"))
    //changetestDf.show(10)

    val testlabelIndexer = new StringIndexer().setInputCol("REASON").setOutputCol("label")
    val testlabelIndexerModel = testlabelIndexer.fit(changetestDf)
    val testdf = testlabelIndexerModel.transform(changetestDf)
    //testdf.show(10)
    //    val df = dfMod.withColumn("label", dfMod("REASON"))
    //    df.printSchema()
    //    df.show(5)

    testdf.createOrReplaceTempView("testaverage")
    //testdf.sqlContext.sql("SELECT avg(cSERVSETD) as avgSERVSETD  FROM testaverage GROUP BY REASON having REASON == 1 ").show

    val testassembler = new VectorAssembler().setInputCols(Array("CASEID", "cSERVSETD", "METHUSE", "cLOS", "SUB1",
      "ROUTE1", "NUMSUBS", "DSMCRIT")).setOutputCol("features")
    val testData = testassembler.transform(testdf)
    //testData.describe("cSERVSETD").show
    //print("below is the test data")
    //testData.show(10)

    val classifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(8)
      .setNumTrees(20)
      .setMaxBins(100)
      .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

    val model = classifier.fit(trainingData)
    println("Random Forest Regresser model: " + model.toDebugString)
    println("model.featureImportances: " + model.featureImportances)

    val predictions = model.transform(testData)
    //predictions.show(10)


    val converter = new IndexToString().setInputCol("prediction")
      .setOutputCol("originalValue")
      .setLabels(labelIndexerModel.labels)
    val df3 = converter.transform(predictions)

    val saveTable = df3.select("CASEID", "cSERVSETD", "METHUSE", "cLOS", "SUB1",
      "ROUTE1", "NUMSUBS", "DSMCRIT", "REASON", "label", "prediction", "originalValue")

    saveTable.coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("/Users/newpc/work/research/treatment/treatment/src/main/Analysis.csv")

    df3.createOrReplaceTempView("Table1")
    df3.sqlContext.sql("select COUNT(distinct CASEID) as count from Table1").show()

    val predictionAndLabels = predictions.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
  }



  def multilayerPerceptronClassifier(dataFrame: DataFrame)={

    val assembler = new VectorAssembler()
      .setInputCols(Array("AGE", "GENDER","RACE","ETHNIC","MARSTAT","EDUC","STFIPS"))
      .setOutputCol("features")


    val df2 = assembler.transform(dataFrame)


    val labelIndexer = new StringIndexer().setInputCol("SUB1").setOutputCol("label")

    val labelIndexerModel = labelIndexer.fit(df2)

    val df3 = labelIndexer.fit(df2).transform(df2)


    // Split the data into train and test
    val splits = df3.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](7, 7, 7, 20)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
      .setSeed(1000) // To make the results reproducible

    // train the model
    val model = trainer.fit(train)

    // compute accuracy on the test set
    val result = model.transform(test)

    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))

  }


  def predictTreatmentCompletion(preOrigDf: DataFrame) = {

    // If the treatment is completed, it is 1, everything else is set to 0
    val column = "REASON"
    val origDf = preOrigDf.withColumn(column, when(col(column).notEqual(1), 0).otherwise(1))
    //origDf.show(10)

    val changeDf = origDf.select(origDf("CASEID"), origDf("SERVSETD") + 1 as "cSERVSETD",origDf("METHUSE"), origDf("LOS")  as "cLOS", origDf("SUB1"),  origDf("ROUTE1"), origDf("NUMSUBS"), origDf("DSMCRIT"), origDf("REASON"))
    //changeDf.show(10)

    val labelIndexer = new StringIndexer().setInputCol("REASON").setOutputCol("label")
    val labelIndexerModel = labelIndexer.fit(changeDf)
    val df = labelIndexerModel.transform(changeDf)
    //df.show(10)

    df.createOrReplaceTempView("average")
    //df.sqlContext.sql("SELECT avg(cSERVSETD) as avgSERVSETD  FROM average GROUP BY REASON having REASON == 1 ").show

    val assembler = new VectorAssembler().setInputCols(Array("CASEID", "cSERVSETD", "METHUSE", "cLOS", "SUB1",
      "ROUTE1", "NUMSUBS", "DSMCRIT")).setOutputCol("features")
    val df2 = assembler.transform(df)
    //df2.describe("cSERVSETD").show
    //df2.show(10)

    val splitSeed = 5043
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), splitSeed)

    val classifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(8)
      .setNumTrees(20)
      .setMaxBins(100)
      .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

    val model = classifier.fit(trainingData)
    println("Random Forest Regresser model: " + model.toDebugString)
    println("model.featureImportances: " + model.featureImportances)

    val predictions = model.transform(testData)
    predictions.show(10)


    val converter = new IndexToString().setInputCol("prediction")
      .setOutputCol("originalValue")
      .setLabels(labelIndexerModel.labels)
    val df3 = converter.transform(predictions)

    df3.select("CASEID","cSERVSETD", "METHUSE", "cLOS", "SUB1",
      "ROUTE1", "NUMSUBS", "DSMCRIT", "REASON", "label", "prediction", "originalValue")
    val predictionAndLabels = predictions.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
  }
}