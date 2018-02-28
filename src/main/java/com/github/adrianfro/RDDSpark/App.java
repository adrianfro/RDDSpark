package com.github.adrianfro.RDDSpark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

public class App {
    
	public static void main( String[] args ) {
		
		final SparkConf sparkConf = new SparkConf();
		sparkConf.set("spark.shuffle.reduceLocality.enabled","false");
		sparkConf.setMaster("local[4]");
		sparkConf.setAppName("BLASSpark");
		//sparkConf.set("spark.memory.useLegacyMode","true");
		
		//The ctx is created from the previous config
        final JavaSparkContext context = new JavaSparkContext(sparkConf);
        //ctx.hadoopConfiguration().set("parquet.enable.summary-metadata", "false");
		
    }
}
