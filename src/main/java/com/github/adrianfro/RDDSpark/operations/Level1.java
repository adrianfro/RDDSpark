package com.github.adrianfro.RDDSpark.operations;

import org.apache.spark.mllib.linalg.DenseVector;

public class Level1 implements Operation {
	
	private static final double ZERO = 0.0;
	
	public static double multiply(DenseVector vector1, DenseVector vector2) {
		
		double result = ZERO;
		
		for (int i = 0; i < vector1.size(); i++) {
			
			result += (vector1.apply(i) * vector2.apply(i));
		}
		
		return result;
	}
	
	public static double sumElements(DenseVector vector) {
		
		double result = ZERO;
		
		for (int i = 0; i < vector.size(); i++) {
			
			result += vector.apply(i);
		}
		
		return result;
	}

}
