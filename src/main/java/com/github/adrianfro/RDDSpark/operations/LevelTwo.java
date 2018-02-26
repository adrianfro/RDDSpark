package com.github.adrianfro.RDDSpark.operations;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.DistributedMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;

import com.github.adrianfro.RDDSpark.spark.MatrixEntriesMultiplication;
import com.github.adrianfro.RDDSpark.spark.MatrixEntriesMultiplicationReductor;

public class LevelTwo implements Operation {
	
	private static final Log LOG = LogFactory.getLog(LevelTwo.class);
	
	/**
	 * generalized matrix-vector multiplication 
	 * y := alpha*A*x + beta*y
	 * 
	 * 
	 */
	
	public static DenseVector dgemv(double alpha, DistributedMatrix A, DenseVector x,
			double beta, DenseVector y, JavaSparkContext context) {
		
		if (beta != 1.0) {
			if (beta == 0.0) {
				y = Vectors.zeros(y.size()).toDense();
			} else {
				BLAS.scal(beta, y);
			}
		}
		
		if (alpha == 0.0) {
			return y;
		}
		
		DenseVector vectorTmp = Vectors.zeros(y.size()).toDense();
		
		// Form y := alpha*A*x + y
		// Case of IndexedRowMatrix
		
		if(A instanceof IndexedRowMatrix) {
			vectorTmp = dgemvIRW((IndexedRowMatrix)A, alpha, x, context);
		} else if (A instanceof CoordinateMatrix) {
			vectorTmp = dgemvCOORD((CoordinateMatrix)A, alpha, x, context);
		} else if (A instanceof BlockMatrix) {
			vectorTmp = dgemvBCK((BlockMatrix)A, alpha, x, context);
		} else {
			vectorTmp = null;
		}
		
		BLAS.axpy(1.0, vectorTmp, y);
		
		return y;
		
	}
	
	private static DenseVector dgemvIRW(IndexedRowMatrix matrix, double alpha, DenseVector vector, JavaSparkContext context) {
		return null;
		
	}
	
	private static DenseVector dgemvCOORD(CoordinateMatrix matrix, double alpha, DenseVector vector, JavaSparkContext context) {
		
		return matrix.entries().toJavaRDD().mapPartitions(new MatrixEntriesMultiplication(vector, alpha))
				.reduce(new MatrixEntriesMultiplicationReductor());
	}
	
	private static DenseVector dgemvBCK(BlockMatrix matrix, double alpha, DenseVector vector, JavaSparkContext context) {
		return null;
	}

}
