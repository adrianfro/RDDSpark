package com.github.adrianfro.RDDSpark;

import org.apache.spark.mllib.linalg.DenseVector;

import com.github.adrianfro.RDDSpark.operations.LevelOne;

public class App {
    
	public static void main( String[] args ) {
		
		double[] values1 = new double[] {1.2, 2.2};
		double[] values2 = new double[] {3.2, 2.4, 3.7};
		
		final DenseVector vector1 = new DenseVector(values1);
		final DenseVector vector2 = new DenseVector(values2);
		
		final double sumElements = LevelOne.sumElements(vector1);
		final double multiplication = LevelOne.multiply(vector1, vector2);
		
		System.out.println("sumElements = " + sumElements);
		System.out.println("multiplication = " + multiplication);
		
    }
}
