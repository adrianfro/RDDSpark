package com.github.adrianfro.RDDSpark.operations;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.distributed.DistributedMatrix;

public class Level3 implements Operation {
	
	private static final Log LOG = LogFactory.getLog(Level3.class);
	
	/**
	 * DGEMM
	 * 
	 * C := alpha * op(A) * op ( B ) + beta * C
	 * 
	 */
	
	public static DistributedMatrix dgemm(double alpha, DistributedMatrix A, DistributedMatrix B, double beta, DistributedMatrix C, JavaSparkContext context) {
		
		
		return null;
	}

}
