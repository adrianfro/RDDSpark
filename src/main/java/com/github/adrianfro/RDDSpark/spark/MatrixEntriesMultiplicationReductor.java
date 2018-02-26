package com.github.adrianfro.RDDSpark.spark;

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;

public class MatrixEntriesMultiplicationReductor implements Function2<DenseVector, DenseVector, DenseVector> {

	private static final long serialVersionUID = 2349557846664005746L;

	public DenseVector call(DenseVector vector1, DenseVector vector2) throws Exception {
		
		BLAS.axpy(1.0, vector1, vector2);
		
		return vector2;
	}

}
