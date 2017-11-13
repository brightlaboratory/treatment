/* SimpleApp.scala */


import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object SimpleApp {

  def createDataframe(spark: SparkSession) = {

    val df = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("data.csv").getPath)

    df.show()


  }
}