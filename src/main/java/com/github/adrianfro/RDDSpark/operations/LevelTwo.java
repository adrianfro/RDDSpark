package com.github.adrianfro.RDDSpark.operations;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.DistributedMatrix;

public class LevelTwo implements Operation {
	
	private static final Log LOG = LogFactory.getLog(LevelTwo.class);
	
	/**
	 * generalized matrix-vector multiplication 
	 * y := alpha*A*x + beta*y
	 * 
	 * 
	 */
	
	public static DenseVector gemv(double alpha, DistributedMatrix A, DenseVector x,
			double beta, DenseVector y, JavaSparkContext context) {
		
		
		
		return y;
		
	}

}
