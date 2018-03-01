package com.github.adrianfro.RDDSpark.operations;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.DistributedMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;

import scala.Tuple2;



public class Level3 implements Operation {
	
	private static final Log LOG = LogFactory.getLog(Level3.class);
	private static final double ZERO = 0.0;
	
	/**
	 * 
	 * C := alpha * op(A) * op ( B ) + beta * C
	 * 
	 * @param alpha
	 * @param A
	 * @param B
	 * @param beta
	 * @param C
	 * @param context
	 * @return
	 */
	
	public static DistributedMatrix dgemm(double alpha, DistributedMatrix A, DistributedMatrix B, double beta, DistributedMatrix C, JavaSparkContext context) {
		
		if (beta != 1.0 && beta != ZERO) {
			//
		}
		
		if (alpha == ZERO) {
			return C;
		}
		
		if(A instanceof IndexedRowMatrix) {
			C = dgemm_IRW((IndexedRowMatrix)A, (IndexedRowMatrix) B, context);
		} else if (A instanceof CoordinateMatrix) {
			
		} else if (A instanceof BlockMatrix) {
			
		} else {
			C = null;
		}
		
		if(alpha != 1.0 && C == null) {
			C = scal(alpha, C, C, context);
		}
		
		
		return C;
	}
	
	// B := alpha * A
	public static DistributedMatrix scal(double alpha, DistributedMatrix A, DistributedMatrix B, JavaSparkContext context){
		
		if (A instanceof IndexedRowMatrix) {
			B = scal_IRW(alpha, (IndexedRowMatrix) A, (IndexedRowMatrix) B, context);
		} else if (A instanceof CoordinateMatrix) {
			B = scal_COORD(alpha, (CoordinateMatrix) A, (CoordinateMatrix) B, context);
		} else if (A instanceof BlockMatrix) {
			B = scal_BCK(alpha, (BlockMatrix) A, (BlockMatrix) B, context);
		} else {
			B = null;
		}
		
		return B;
	}
	
	@SuppressWarnings("serial")
	private static IndexedRowMatrix scal_IRW(double alpha, IndexedRowMatrix A, IndexedRowMatrix B, JavaSparkContext context) {

        final JavaRDD<IndexedRow> rows = A.rows().toJavaRDD();

        final Broadcast<Double> alphaBroadcast = context.broadcast(alpha);

        final JavaRDD<IndexedRow> newRows = rows.map(new Function<IndexedRow, IndexedRow>() {
        	
            public IndexedRow call(IndexedRow indexedRow) throws Exception {

                final double alphaValue = alphaBroadcast.getValue().doubleValue();

                final long index = indexedRow.index();

                final double values[] = new double[indexedRow.vector().size()];

                for(int i = 0; i< values.length; i++) {
                    values[i] = indexedRow.vector().apply(i) * alphaValue;
                }

                return new IndexedRow(index, new DenseVector(values));

            }
        });

        return new IndexedRowMatrix(newRows.rdd());

	}
	
	@SuppressWarnings("serial")
	private static CoordinateMatrix scal_COORD(double alpha, CoordinateMatrix A, CoordinateMatrix B, JavaSparkContext context) {

        final JavaRDD<MatrixEntry> entries = A.entries().toJavaRDD();
        final Broadcast<Double> alphaBroadcast = context.broadcast(alpha);

        final JavaRDD<MatrixEntry> newEntries = entries.mapPartitions(new FlatMapFunction<Iterator<MatrixEntry>, MatrixEntry>() {
            
        	public Iterator<MatrixEntry> call(Iterator<MatrixEntry> matrixEntryIterator) throws Exception {

                final double alphaValue = alphaBroadcast.getValue().doubleValue();
                final List<MatrixEntry> newEntries = new ArrayList<MatrixEntry>();

                while(matrixEntryIterator.hasNext()) {
                	
                    final MatrixEntry currentEntry = matrixEntryIterator.next();
                    newEntries.add(new MatrixEntry(currentEntry.i(), currentEntry.j(), currentEntry.value() * alphaValue));

                }
                return newEntries.iterator();
            }
        });

        return new CoordinateMatrix(newEntries.rdd());

	}
	
	@SuppressWarnings("serial")
	private static BlockMatrix scal_BCK(double alpha, BlockMatrix A, BlockMatrix B, JavaSparkContext context) {

        final JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> blocks = A.blocks().toJavaRDD();
        final Broadcast<Double> alphaBroadcast = context.broadcast(alpha);

        final JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> newBlocks = blocks.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {
           
        	public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> block) throws Exception {

                final double alphaBroadcastRec = alphaBroadcast.getValue().doubleValue();
                final Integer row = (Integer)block._1._1; //Integer.parseInt(block._1._1.toString());
                final Integer col = (Integer)block._1._2;
                final Matrix matrixBlock = block._2;

                for(int i = 0; i< matrixBlock.numRows(); i++) {

                    for(int j = 0; j< matrixBlock.numCols(); j++) {
                        matrixBlock.update(i,j, matrixBlock.apply(i,j) * alphaBroadcastRec);
                    }

                }

                return new Tuple2<Tuple2<Object, Object>, Matrix>(new Tuple2<Object, Object>(row, col), matrixBlock);

            }
        });

        return new BlockMatrix(newBlocks.rdd(), A.rowsPerBlock(), A.colsPerBlock());

	}
	
	public static IndexedRowMatrix dgemm_IRW(IndexedRowMatrix A, IndexedRowMatrix B, JavaSparkContext context){

        /*IndexedRow[] rowsB = B.rows().collect();
        final Broadcast<IndexedRow[]> rowsBC = jsc.broadcast(rowsB);
        JavaRDD<IndexedRow> newRows = A.rows().toJavaRDD().map(new Function<IndexedRow, IndexedRow>() {
            @Override
            public IndexedRow call(IndexedRow indexedRow) throws Exception {
                IndexedRow[] rowsBValues = rowsBC.getValue();
                double newRowValues[] = new double[rowsBValues[0].vector().size()];
                for(int i = 0; i< newRowValues.length ; i++) {
                    newRowValues[i] = 0.0;
                }
                Vector vect = indexedRow.vector();
                for(int i = 0; i< rowsBValues.length; i++) {
                    newRowValues[i] = BLAS.dot(vect, rowsBValues[i].vector());
                }
                Vector result = new DenseVector(newRowValues);
                return new IndexedRow(indexedRow.index(), result);
            }
        });
        return new IndexedRowMatrix(newRows.rdd());*/

        return A.multiply(toLocal(B));

    }
	
	public static CoordinateMatrix DGEMM_COORD( CoordinateMatrix A, CoordinateMatrix B, JavaSparkContext context){
        // Temporal method
        return dgemm_IRW(A.toIndexedRowMatrix(), B.toIndexedRowMatrix(), context).toCoordinateMatrix();
    }

    public static BlockMatrix DGEMM_BCK(BlockMatrix A, BlockMatrix B, JavaSparkContext context) {
        return dgemm_IRW(A.toIndexedRowMatrix(), B.toIndexedRowMatrix(), context).toBlockMatrix();
    }
	
	private static Matrix toLocal(IndexedRowMatrix A) {

        final int numRows = (int)A.numRows();
        final int numCols = (int)A.numCols();
        final List<IndexedRow> rows = A.rows().toJavaRDD().collect();

        final Vector vectors[] = new Vector[rows.size()];

        for(int i = 0; i< rows.size(); i++) {
            vectors[(int)rows.get(i).index()] = rows.get(i).vector();
        }
        
        final double values[] = new double[numRows * numCols];

        for(int i = 0; i< numRows; i++) {

            for(int j = 0; j < numCols; j++) {

                values[j * numRows + i] = vectors[i].apply(j);
            }
        }

        return new DenseMatrix(numRows, numCols, values);


	}
	
	private static Matrix toLocal(CoordinateMatrix A) {

        final int numRows = (int)A.numRows();
        final int numCols = (int)A.numCols();
        final List<MatrixEntry> entries = A.entries().toJavaRDD().collect();

        final double values[] = new double[numRows * numCols];

        for(MatrixEntry currentEntry: entries) {
            values[(int)(currentEntry.j() * numRows + currentEntry.i())] = currentEntry.value();
        }

        return new DenseMatrix(numRows, numCols, values);
	}

}
