package com.citi.transformations

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StringType, StructField, StructType}

//Clase de un discretizador equalDepth
class EqualCountBinner(SparkSession_Param:SparkSession) extends Binner{
  var counter:Int=3
  var colToBin=""
  var arrRange=Array.fill(0)(0.0)
  var splitSize=0
  var OutputColName="Col_Bin"
  val spark=SparkSession_Param

  override def setNumBuckets(countBin:Int=3):this.type ={
    this.counter=countBin
  this
  }

  override def setInputColName(Col:String):this.type={
    this.colToBin=Col
    this
  }

  override def setOutputColName(Col:String):this.type={
    this.OutputColName=Col
    this
  }

  override def fit(train:Dataset[Row]):this.type={
    this.splitSize=(train.count()/this.counter).toInt
    this
  }

  override def transform(transf:Dataset[Row]):Dataset[Row]={
    val header=transf.schema.add(StructField(OutputColName,StringType))
    val w = Window.orderBy(colToBin)
    transf.sort(colToBin).withColumn("index", row_number().over(w))
      .withColumn(OutputColName,binner(col("index"))).drop("index1")
  }

  private def binnerFunc: Any => String = { id => var bin=((id.toString.toDouble-1)/splitSize).toInt
    if(bin<this.counter)
      bin+"_Bin_"+OutputColName
    else
      bin-1+"_Bin_"+OutputColName}

  private val binner = udf(binnerFunc)

}
