package com.citi.transformations

import org.apache.spark.sql.{Dataset, DatasetHolder, Row}

//Clase abstracta basica de un discretizador
trait Binner extends Serializable{

  protected def setInputColName(Col: String):Binner=this
  protected def setOutputColName(Col: String):Binner=this
  protected def setNumBuckets(Num:Int=3): Binner =this
  protected def fit(train:Dataset[Row]):Binner=this
  protected def transform(transf:Dataset[Row]):Dataset[Row]=transf
}
