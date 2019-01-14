package sample

import org.junit.jupiter.api.Test
import java.time.Duration
import kotlin.system.measureTimeMillis

internal class SomeDLTest {

    @Test
    fun someTest() {
        val dur = measureTimeMillis {
            val trainer = SomeDL()
            trainer.main()
        }

        println("Duration ${Duration.ofMillis(dur)}")
    }

}