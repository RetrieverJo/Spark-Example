package com.hyunje.jo.spark;

import org.apache.commons.cli.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.*;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import java.util.ArrayList;
import java.util.List;

public class SparkMatrixTest {
    public static final String delim = " ";

    public static void main(String[] args) throws ParseException {
        String INPUT_PATH = "", OUTPUT_PATH = "";

        //Commandline Parsing
        Options options = new Options();
        options.addOption("i", "input", true, "input path(HDFS)");
        options.addOption("o", "output", true, "output path(HDFS)");

        CommandLineParser parser = new BasicParser();
        CommandLine cmd = parser.parse(options, args);
        if (cmd.hasOption("i")) {
            INPUT_PATH = cmd.getOptionValue("i");
        } else {
            System.err.println("Input path is invalid");
        }

        if (cmd.hasOption("o")) {
            OUTPUT_PATH = cmd.getOptionValue("o");
        } else {
            System.err.println("Output path is invalid");
        }

        SparkConf conf = new SparkConf().setAppName("Spark-matrix-test").setMaster("local");
        JavaSparkContext context = new JavaSparkContext(conf);


        //테스트용 Matrix 생성하는 과정
        JavaRDD<String> data = context.textFile(INPUT_PATH);
        JavaRDD<Vector> valueVector = data.map(
                new Function<String, Vector>() {
                    public Vector call(String s) {
                        String[] sarray = s.split(delim);
                        double[] darray = new double[sarray.length];
                        for (int i = 0; i < sarray.length; i++) {
                            darray[i] = Double.parseDouble(sarray[i]);
                        }
                        return Vectors.dense(darray);
                    }
                }
        );

        //Singular Value Decomposition 수행
        RowMatrix matrix = new RowMatrix(valueVector.rdd());
        SingularValueDecomposition<RowMatrix, Matrix> svd = matrix.computeSVD(2, true, 1.0E-9d);
        RowMatrix U = svd.U();
        Vector s = svd.s();
        Matrix V = svd.V();

        //입출력 테스트용 매트릭스
        Matrix vTransposed = V.transpose();

        //매트릭스 내용 출력
        System.out.println("rows: " + vTransposed.numRows());
        System.out.println("cols: " + vTransposed.numCols());
        for (int r = 0; r < vTransposed.numRows(); r++) {
            for (int c = 0; c < vTransposed.numCols(); c++) {
                System.out.print(vTransposed.apply(r, c) + " ");
            }
            System.out.println();
        }

        //매트릭스를 RDD로 변환하여 RDD[Marix]를 ObjectFile로 저장
        System.out.println("Save as Object File!!");
        List<Matrix> matrixList = new ArrayList<Matrix>();
        matrixList.add(vTransposed);
        context.parallelize(matrixList).saveAsObjectFile(OUTPUT_PATH);

        //저장한 Object File을 RDD[Matrix] 형태로 불러옴
        System.out.println("Load Object File");
        JavaRDD<Matrix> loadedMatrixRdd = context.objectFile(OUTPUT_PATH);

        //불러온 RDD[Matrix]중 제일 첫번째 Matrix만 가져옴
        Matrix loadedMatrix = loadedMatrixRdd.first();

        //정상적으로 불러왔는지 출력
        System.out.println("rows: " + loadedMatrix.numRows());
        System.out.println("cols: " + loadedMatrix.numCols());
        for (int r = 0; r < loadedMatrix.numRows(); r++) {
            for (int c = 0; c < loadedMatrix.numCols(); c++) {
                System.out.print(loadedMatrix.apply(r, c) + " ");
            }
            System.out.println();
        }

        context.stop();
    }
}
