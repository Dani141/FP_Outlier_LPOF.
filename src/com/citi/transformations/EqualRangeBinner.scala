package com.citi.transformations

import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
//Clase de un Discretizador equalWidth
class EqualRangeBinner extends Binner {
  var counter:Int=3
  var maxValue=0.0
  var minValue=0.0
  var colToBin=""
  var splitSize=0.0
  var OutputColName="Col_Bin"

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
    val arrData=train.describe(colToBin).drop("summary").collect().slice(3,5)
    this.maxValue=arrData(1).getString(0).toDouble
    this.minValue=arrData(0).getString(0).toDouble
    this.splitSize=(this.maxValue-this.minValue)/this.counter
    this
  }

  override def transform(transf:Dataset[Row]):Dataset[Row]={
  transf.withColumn(OutputColName,binner(col(colToBin)))
  }

  private def binnerFunc: Any => String = { num => var bin=((num.toString.toDouble-this.minValue)/splitSize).toInt
    if(bin<this.counter)
    bin+"_Bin_"+OutputColName
    else
    bin-1+"_Bin_"+OutputColName}

  private val binner = udf(binnerFunc)

}


