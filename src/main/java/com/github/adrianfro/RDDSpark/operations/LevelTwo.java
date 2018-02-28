package com.github.adrianfro.RDDSpark.operations;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.DistributedMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;

import com.github.adrianfro.RDDSpark.spark.MatrixEntriesMultiplication;
import com.github.adrianfro.RDDSpark.spark.MatrixEntriesMultiplicationReductor;

import scala.Tuple2;

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
	
	@SuppressWarnings("serial")
	private static DenseVector dgemvIRW(IndexedRowMatrix matrix, double alpha, DenseVector vector, JavaSparkContext context) {
		
		final Broadcast<DenseVector> broadcast = context.broadcast(vector);
		final Broadcast<Double> alphaBroadcast = context.broadcast(alpha);
		
		final JavaRDD<IndexedRow> rows = matrix.rows().toJavaRDD();
		final List<Tuple2<Long, Double>> returnValues = rows.mapToPair(new PairFunction<IndexedRow, Long, Double>() {

			public Tuple2<Long, Double> call(IndexedRow row) throws Exception {
				
				final DenseVector vector = broadcast.getValue();
				final double alphaBroadcastRec = alphaBroadcast.getValue();
				
				final DenseVector vectorTmp = row.vector().copy().toDense();
				
				BLAS.scal(alphaBroadcastRec, vectorTmp);
				
				return new Tuple2<Long, Double>(row.index(), BLAS.dot(vectorTmp, vector));
			}

		}).collect();
		
		final double[] stockArray = new double[returnValues.size()];
		
		for (final Tuple2<Long, Double> tuple : returnValues) {
			stockArray[tuple._1().intValue()] = tuple._2();			
		}
		
		return new DenseVector(stockArray);
		
	}
	
	private static DenseVector dgemvCOORD(CoordinateMatrix matrix, double alpha, DenseVector vector, JavaSparkContext context) {
		
		return matrix.entries().toJavaRDD().mapPartitions(new MatrixEntriesMultiplication(vector, alpha))
				.reduce(new MatrixEntriesMultiplicationReductor());
	}
	
	@SuppressWarnings("serial")
	private static DenseVector dgemvBCK(BlockMatrix matrix, double alpha, DenseVector vector, JavaSparkContext context) {
		
		final Broadcast<DenseVector> broadcast = context.broadcast(vector);
		final Broadcast<Double> alphaBroadcast = context.broadcast(alpha);
		
		// Theoretically, the index should be a Tuple2<Integer, Integer>
        final JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> blocks = matrix.blocks().toJavaRDD();
//        JavaRDD<Tuple2<Tuple2<Integer, Integer>, Matrix>> blocks = matrix.blocks().toJavaRDD();
        
        final DenseVector returnValues = blocks.map(new Function<Tuple2<Tuple2<Object,Object>,Matrix>, DenseVector>() {

			public DenseVector call(Tuple2<Tuple2<Object, Object>, Matrix> block) throws Exception {
				
				LOG.debug("Entering Map Phase");
				final DenseVector inputVector = broadcast.getValue();
				final double alphaBroadcastVector = alphaBroadcast.getValue();
				
				LOG.debug("Vector items : " + inputVector.size());
				final double finalResultArray[] = new double[inputVector.size()];
				Arrays.fill(finalResultArray, 0.0);
				
				LOG.debug("Before loading rows and columns: " + inputVector.size());
				Integer row = (Integer)block._1._1; //Integer.parseInt(block._1._1.toString());
                Integer col = (Integer)block._1._2;//Integer.parseInt(block._1._2.toString());

                LOG.debug("Row is: " + row);
                LOG.debug("Col is: " + col);
                
                Matrix matrix = block._2;

                double vectValues[] = new double[matrix.numCols()];
                double resultArray[] = new double[matrix.numCols()];

                for(int i = col; i < matrix.numCols();i++) {
                    vectValues[(i-col)] = inputVector.apply(i);
                    resultArray[(i-col)] = 0.0;
                }

                final DenseVector result = Vectors.zeros(matrix.numCols()).toDense();//new DenseVector(resultArray);

                BLAS.gemv(alphaBroadcastVector, matrix, new DenseVector(vectValues), 0.0, result);

                for(int i = col; i < matrix.numCols();i++) {
                    finalResultArray[i] = result.apply((i-col));
                }

                return new DenseVector(finalResultArray);
			}
		}).reduce(new Function2<DenseVector, DenseVector, DenseVector>() {
			
			public DenseVector call(DenseVector vector1, DenseVector vector2) throws Exception {
				
				final double result[] = new double[vector1.size()];
				
				for (int i = 0; i < result.length; i++) {
					result[i] = vector1.apply(i) + vector2.apply(i);
				}
				
				return new DenseVector(result);
			}
		});
        
		return returnValues;
	}

}
