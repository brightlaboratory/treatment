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

    val df_new = df.select("AGE", "GENDER", "STFIPS", "SERVSETD", "NOPRIOR", "SUB1", "SUB2", "SUB3")

    val df_label=df.select("AGE","GENDER","RACE","EDUC","EMPLOY")
    // Convert all columns into integer
    val someCastedDF = (df_label.columns.toBuffer).foldLeft(df_label)((current, c) =>current.withColumn(c, col(c).cast("int")))
    someCastedDF.printSchema()

    //kMeansClustering(someCastedDF)
    calculateColumnValuePercentage(someCastedDF)


  }
  //Calculate percentage of each unique range in a column

  def calculateColumnValuePercentage(dataFrame: DataFrame)={

    val total_records= dataFrame.count()
    print(total_records)


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
    def func_percentAge = udf((age:Int, count: Int) => {
      if (age == 2)      (count/1048575.0)*100
      else if (age == 3) (count/1048575.0)*100
      else if (age == 4) (count/1048575.0)*100
      else if (age == 5) (count/1048575.0)*100
      else if (age == 6) (count/1048575.0)*100
      else if (age == 7) (count/1048575.0)*100
      else if (age == 8) (count/1048575.0)*100
      else if (age == 9) (count/1048575.0)*100
      else if (age == 10) (count/1048575.0)*100
      else if (age == 11) (count/1048575.0)*100
      else if (age == 12) (count/1048575.0)*100
      else 0
    })

    val new_df1 = ageDF.withColumn("percentage", func_percentAge(col("AGE"),col("count")))

    print("Final df of AGE with percentage is:\n")
    new_df1.show()

    /*val outputFile="ageData"
    new_df1.write.option("header", "true").csv(outputFile)*/
    new_df1.coalesce(1).write.option("header", "true").csv("/Users/anujatike/Documents/sem3/RA/Project2/HistogramData/ageData.csv")
  }


  def assignOriginalColumnLabelsAGE(dataFrame: DataFrame)= {

    dataFrame.show()

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
    new_df1.show()

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
    new_df2.show()
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

    new_df3.show()
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

}