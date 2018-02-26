package com.github.adrianfro.RDDSpark.spark;

import java.util.Arrays;
import java.util.Iterator;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;

public class MatrixEntriesMultiplication implements FlatMapFunction<Iterator<MatrixEntry>, DenseVector> {

	private static final long serialVersionUID = -4352074036673843751L;
	private static final double ZERO = 0.0;
	
	private final DenseVector vector;
	private final double alpha;

	public MatrixEntriesMultiplication(DenseVector vector, double alpha) {
		this.vector = vector;
		this.alpha = alpha;
	}

	public Iterator<DenseVector> call(Iterator<MatrixEntry> matrixIterator) throws Exception {
		
		final double result[] = new double[vector.size()];
		Arrays.fill(result, ZERO); 
		
		while (matrixIterator.hasNext()) {
			final MatrixEntry entry = matrixIterator.next();
			//TODO No me cuadra (esos cast ...)
			result[(int)entry.i()] += (this.vector.apply((int)entry.j()) * entry.value() * this.alpha);
		}
		
		return Arrays.asList(new DenseVector(result)).iterator();
	}

}
