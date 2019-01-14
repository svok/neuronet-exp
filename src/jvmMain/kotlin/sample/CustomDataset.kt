package sample

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File


class CustomDataset(
    val batchSize: Int = 150,
    val labelIndex: Int = 4,
    val numClasses: Int = 3,
    val numInputs: Int = 4,
    val outputNum: Int = 3,
    val seed: Long = 6L,
    val fileName: String = "custom-dataset.bin"
) {

    private val dataSetIterator = prepareData().next()!!.apply {
        shuffle()
    }
    private val testAndTrain = dataSetIterator.splitTestAndTrain(0.75)!!;  //Use 65% of data for training

    private val trainingData = testAndTrain.train!!
    private val testData = testAndTrain.test!!
    private val eval = Evaluation(3)
    val normalizer = NormalizerStandardize().apply {
        fit(trainingData)           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        transform(trainingData)     //Apply normalization to the training data
        transform(testData)         //Apply normalization to the test data. This is using statistics calculated from the *training* set
    }

    private var trainingRequired = true

    val model = try {
//        createNN()
        val nn = loadNN()
        trainingRequired = false
        nn
    } catch (e: Exception) {
        createNN()
    }

    init {
        if (trainingRequired) train()
    }

    private fun prepareData(): RecordReaderDataSetIterator {
        val recordReader = CSVRecordReader(0, ',', '\'').apply {
//            initialize(FileSplit(ClassPathResource("iris.txt", this.javaClass.classLoader).file))
            initialize(FileSplit(File("iris.txt")))
        }
        return RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses)
    }

    private fun loadNN(): MultiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(File(fileName))

    private fun createNN(): MultiLayerNetwork {
        val conf = NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(Sgd(0.1))
            .l2(1e-4)
            .list()
            .layer(0, DenseLayer
                .Builder()
                .nIn(numInputs)
                .nOut(15)
                .build()
            )
            .layer(1, DenseLayer
                .Builder()
                .nIn(15)
                .nOut(15)
                .build()
            )
            .layer(2, OutputLayer
                .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(15)
                .nOut(outputNum)
                .build()
            )
            .backpropType(BackpropType.Standard)
            .pretrain(true)
            .build();

        //run the model
        return MultiLayerNetwork(conf).apply {
            init()
            setListeners(ScoreIterationListener(100))
        }

    }

    fun train() {
        for(i in 0 until 1000 ) {
            val x = trainingData
//            println("ZZ: $x")
            model.fit(x)
        }

        //evaluate the model on the test set

        val output = model.output(testData.features)!!
        val mx = testData.withIndex().map { (i, v) ->
            arrayOf(
                *v.features.toDoubleVector().toTypedArray(),
                *v.labels.toDoubleVector().toTypedArray(),
                *output.getRow(i.toLong()).toDoubleVector().toTypedArray()
            )
        }
        mx.forEach {
            it.forEach { d: Double ->
                print("\t$d")
            }
            println()
        }

        eval.eval(testData.labels, output)
//        println("EVAL: ${eval.stats()}")

        model.save(File("custom-dataset.bin"))
    }

    fun eval(data: INDArray): INDArray {
        normalizer.transform(data)
        return model.output(data, false)
    }

}