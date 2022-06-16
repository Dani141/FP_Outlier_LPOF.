package main.samples

import com.citi.ml.FP_Outlier
import com.citi.transformations.EqualRangeBinner
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.functions.desc
import org.apache.spark.storage.StorageLevel;
//Main class que prepara los resultadso para ser evaluados en ROC
object mainHttp {
  def main(args: Array[String]): Unit = {
    val spark=SparkSession.builder().master("local[8]").appName("TEST").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._
    val support=0.05
    val masterPathout="data/output/smtp"+support+"/"
    val input="data/dataset/smtp.csv"
    var data = spark.read.option("header","true").option("inferSchema","true").csv(input).repartition(8)
      //.withColumn("ID",monotonically_increasing_id())

    val header=data.columns.filter(x=>(!x.contains("ID"))&&(!x.contains("Class")))
    header.foreach{x=>
      data=new EqualRangeBinner()
        .setNumBuckets(6)
        .setInputColName(x)
        .setOutputColName(x+"_bin")
        .fit(data)
        .transform(data)
    }

    var algLFPOF= new FP_Outlier()
      .setMinConfidence(0.7)
      .setMinSupport(support)
      .setColumns(data.columns
        .filter(x=>x.contains("_bin")))
      .train(data)

    val original=algLFPOF.transform(data,spark).persist(StorageLevel.MEMORY_AND_DISK)
    original.count()
    saveDataset(original.drop("features"),masterPathout+"fulldata")

    val sortWCPOF=original.sort("WCPOF_METRIC").select("ID","Class","WCPOF_METRIC")
    saveDataset(sortWCPOF,masterPathout+"sortDescWCPOF")

    val sortLFPOF=original.sort("LFPOF_METRIC").select("ID","Class","LFPOF_METRIC")
    saveDataset(sortLFPOF,masterPathout+"sortDescLFPOF")

    val sorFPOF=original.sort("FPOF_METRIC").select("ID","Class","FPOF_METRIC")
    saveDataset(sorFPOF,masterPathout+"sortDescFPOF")

  }

  def saveDataset(data:Dataset[Row], path:String)={
    data.coalesce(1).write.option("header","true").mode("overwrite").csv(path)
  }
}
