package sample

import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j

internal class CustomDatasetTest {

    @Test
    fun dataset() {
        val dataset = CustomDataset()
//        dataset.train()

//        val dataIn = Nd4j.create(
//            arrayOf(4.7,3.2,1.3,0.2, 5.6,3.0,4.1,1.3, 6.0,3.0,4.8,1.8).toDoubleArray(),
//            arrayOf(3, 4).toIntArray(),
//            'c'
//        )
//        println("RESULT0 = ${dataset.eval(dataIn)}")

        val dataIn0 = Nd4j.create(arrayOf(4.7,3.2,1.3,0.2).toDoubleArray())
        println("RESULT0 = ${dataset.eval(dataIn0)}")
        val dataIn1 = Nd4j.create(arrayOf(5.6,3.0,4.1,1.3).toDoubleArray())
        println("RESULT1 = ${dataset.eval(dataIn1)}")
        val dataIn2 = Nd4j.create(arrayOf(6.4,2.8,5.6,2.1).toDoubleArray())
        println("RESULT2 = ${dataset.eval(dataIn2)}")
    }

}