/* SimpleApp.scala */


import org.apache.spark.SparkContext

object SimpleApp {

  def countWords(sc: SparkContext): Unit = {
    val pathToFiles = "/Users/newpc/work/research/healthsage/pom.xml"
    val files = sc.textFile(pathToFiles)
    println("Count: " + files.count())
  }
}