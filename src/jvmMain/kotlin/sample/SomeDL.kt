package sample

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.evaluation.classification.ROCMultiClass
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.concurrent.TimeUnit


class SomeDL {
    var batchSize = 16 // how many examples to simultaneously train in the network
    var emnistSet: EmnistDataSetIterator.Set = EmnistDataSetIterator.Set.MNIST
    var rngSeed = 123
    var numRows = 28
    var numColumns = 28
    var reportingInterval = 5

    fun main() {
        println("starting!!!")
        // create the data iterators for emnist
        val emnistTrain = EmnistDataSetIterator(emnistSet, batchSize, true)
        val emnistTest = EmnistDataSetIterator(emnistSet, batchSize, false)

        val outputNum = EmnistDataSetIterator.numLabels(emnistSet)

        // network configuration (not yet initialized)
        val conf = NeuralNetConfiguration.Builder()
            .seed(rngSeed.toLong())
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Adam())
            .l2(1e-4)
            .list()
            .layer(
                DenseLayer.Builder()
                    .nIn(numRows * numColumns) // Number of input datapoints.
                    .nOut(1000) // Number of output datapoints.
                    .activation(Activation.RELU) // Activation function.
                    .weightInit(WeightInit.XAVIER) // Weight initialization.
                    .build()
            )
            .layer(
                DenseLayer.Builder()
                    .nIn(1000)
                    .nOut(1000)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build()
            )
            .layer(
                DenseLayer.Builder()
                    .nIn(1000)
                    .nOut(1000)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(1000)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build()
            )
            .pretrain(false)
            .backpropType(BackpropType.Standard)
            .build()

        println("created")
        // create the MLN
        val network = MultiLayerNetwork(conf)
        network.init()
        println("network initialized")

        // pass a training listener that reports score every N iterations
        network.addListeners(ScoreIterationListener(reportingInterval))

        // here we set up an early stopping trainer
        // early stopping is useful when your trainer runs for
        // a long time or you need to programmatically stop training
        val esConf = EarlyStoppingConfiguration
            .Builder<MultiLayerNetwork>()
            .epochTerminationConditions(MaxEpochsTerminationCondition(5))
            .iterationTerminationConditions(MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
            .scoreCalculator(DataSetLossCalculator(emnistTest, true))
            .evaluateEveryNEpochs(1)
            .modelSaver(LocalFileModelSaver(System.getProperty("user.dir")))
            .build()

        println("esConf")
        // training
        val trainer = EarlyStoppingTrainer(esConf, network, emnistTrain)
        println("training started")
        val result = trainer.fit()
        println("training done")

        // print out early stopping results
        println("Termination reason: " + result.terminationReason)
        println("Termination details: " + result.terminationDetails)
        println("Total epochs: " + result.totalEpochs)
        println("Best epoch number: " + result.bestModelEpoch)
        println("Score at best epoch: " + result.bestModelScore)

        // evaluate basic performance
        val eval = network.evaluate<Evaluation>(emnistTest)
        System.out.println(eval.accuracy())
        System.out.println(eval.precision())
        System.out.println(eval.recall())

        // evaluate ROC and calculate the Area Under Curve
        val roc = network.evaluateROCMultiClass<ROCMultiClass>(emnistTest)
        System.out.println(roc.calculateAverageAUC())

        // calculate AUC for a single class
        val classIndex = 0
        System.out.println(roc.calculateAUC(classIndex))

        // optionally, you can print all stats from the evaluations
        System.out.println(eval.stats())
        System.out.println(roc.stats())
    }
}
