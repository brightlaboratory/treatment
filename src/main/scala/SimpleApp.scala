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

    //df.show()

    val df_new = df.select("AGE", "GENDER", "STFIPS", "SERVSETD", "NOPRIOR", "SUB1", "SUB2", "SUB3")


    // Convert all columns into integer
    //val someCastedDF = (df_new.columns.toBuffer).foldLeft(df_new)((current, c) =>current.withColumn(c, col(c).cast("int")))
    //kMeansClustering(someCastedDF)

    // Selecting columns to pass to the assignOriginalColumnLabels function

    val df_label=df.select("AGE","GENDER","RACE","EDUC","EMPLOY")
    val someCastedDF = (df_label.columns.toBuffer).foldLeft(df_label)((current, c) =>current.withColumn(c, col(c).cast("int")))
    someCastedDF.printSchema()
    assignOriginalColumnLabels(someCastedDF)


  }


  def assignOriginalColumnLabels(dataFrame: DataFrame)={

    dataFrame.show()

    // Adding Actual Name Labels to column AGE

    val func_age: (Int => String) = (age: Int) => {
      if (age == 2) "12-14"
      else if (age==3) "15-17"
      else if (age==4) "18-20"
      else if (age==5) "21-24"
      else if (age==6) "25-29"
      else if (age==7) "30-34"
      else if (age==8) "35-39"
      else if (age==9) "40-44"
      else if (age==10) "45-49"
      else if (age==11) "50-54"
      else if (age==12) "55 AND OVER"
      else "MISSING/UNKNOWN/NOT COLLECTED/INVALID"
    }
    val func1 = udf(func_age)

    val new_df1=dataFrame.withColumn("originalAGELabels", func1(col("AGE")))

    // Adding Actual Name Labels to column GENDER

    val func_gender: (Int => String) = (gender: Int) => {
      if (gender == 1) "MALE"
      else if (gender==2) "FEMALE"
      else "MISSING/UNKNOWN/NOT COLLECTED/INVALID"
    }
    val func2 = udf(func_gender)
    val new_df2=new_df1.withColumn("originalGENDERLabels", func2(col("GENDER")))

   //Adding Actual Name Labels to column AGE

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
    val new_df3=new_df2.withColumn("originalRACELabels", func3(col("RACE")))

    new_df3.show()

    //Calculate percentage of each unique range in a column
    calculateColumnValuePercentage(new_df3)


  }

  //Calculate percentage of each unique range in a column

  def calculateColumnValuePercentage(dataFrame: DataFrame)={

   val total_records= dataFrame.count()
    print(total_records)
/*
    dataFrame.createOrReplaceTempView("tempView")
    val count_age_2 = dataFrame.sparkSession.sql("SELECT COUNT(AGE)" + "FROM tempView " +
      "WHERE AGE==2")
    print(count_age_2.dtypes)*/


    // Count of unique values in columns
    val countAge=dataFrame.groupBy("AGE").count()
    val countGENDER=dataFrame.groupBy("GENDER").count()
    val countRace=dataFrame.groupBy("RACE").count()

    // Join counts with df
    val joinedAGEdf=joinOnAGE(dataFrame,countAge)
    val joinedGENDERdf=joinOnGENDER(joinedAGEdf,countGENDER)
    val joinedFinaldf=joinOnRACE(joinedGENDERdf,countRace)


    //joinedAGEdf.show()
    //joinedGENDERdf.show()
    joinedFinaldf.show()



  }

  //Joining count of each unique column value with original dataframe

  def joinOnAGE(df1: DataFrame, df2: DataFrame) = {

    val joinedDf = df1.join(df2, df1("AGE") === df2("AGE"), "inner")
    joinedDf
  }

  def joinOnGENDER(df1: DataFrame, df2: DataFrame) = {

    val joinedDf = df1.join(df2, df1("GENDER") === df2("GENDER"), "inner")
    joinedDf
  }

  def joinOnRACE(df1: DataFrame, df2: DataFrame) = {

    val joinedDf = df1.join(df2, df1("RACE") === df2("RACE"), "inner")
    joinedDf
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