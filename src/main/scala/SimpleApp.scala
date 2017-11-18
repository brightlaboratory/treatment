/* SimpleApp.scala */


import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}



object SimpleApp {

  def createDataframe(spark: SparkSession) = {

    val df = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("data.csv").getPath)

    df.show()

    val df_new = df.select("AGE", "GENDER", "STFIPS", "SERVSETD", "NOPRIOR", "SUB1", "SUB2", "SUB3")
    df_new.printSchema()
    df_new.createOrReplaceTempView("kmeans")
 /*
    val final_df= spark.sql("SELECT * FROM kmeans WHERE AGE!='-9' OR GENDER!='-9' " +
      "OR STFIPS !='-9' OR SERVSETD!='-9' OR NOPRIOR!='-9'OR SUB1!='-9' OR SUB2!='-9' OR SUB3!='-9'")*/

    //val final_df= spark.sql("DELETE FROM kmeans WHERE GENDER=='-9'")






    // Changing datatypes of columns from string to integre

    val someCastedDF = (df_new.columns.toBuffer).foldLeft(df_new)((current, c) =>current.withColumn(c, col(c).cast("int")))

    someCastedDF.printSchema()



    kMeansClustering(someCastedDF)

  }

  def kMeansClustering(dataFrame: DataFrame)= {

    /*
    val encoder = new OneHotEncoder().setInputCol("AGE").setOutputCol("AGEVec")
    val encoder2 = new OneHotEncoder().setInputCol("GENDER").setOutputCol("GENDERVec")
    val assembler = new VectorAssembler().setInputCols(Array("AGEVec","GENDERVec")).setOutputCol("features")
    val kmeans = new KMeans().setK(2).setFeaturesCol("features").setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(encoder,encoder2, assembler, kmeans))

    val kMeansPredictionModel = pipeline.fit(dataFrame)

    val predictionResult = kMeansPredictionModel.transform(dataFrame)*/

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

}