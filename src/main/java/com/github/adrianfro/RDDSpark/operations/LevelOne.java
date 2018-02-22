package com.github.adrianfro.RDDSpark.operations;

import org.apache.spark.mllib.linalg.DenseVector;

public class LevelOne implements Operation {
	
	public static double multiply(DenseVector vector1, DenseVector vector2) {
		
		double result = 0.0;
		
		for (int i = 0; i < vector1.size(); i++) {
			
			result += (vector1.apply(i) * vector2.apply(i));
		}
		
		return result;
	}
	
	public static double sumElements(DenseVector vector) {
		
		double result = 0.0;
		
		for (int i = 0; i < vector.size(); i++) {
			
			result += vector.apply(i);
		}
		
		return result;
	}

}
