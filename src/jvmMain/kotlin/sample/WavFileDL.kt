package sample

import org.datavec.api.records.reader.RecordReader
import org.datavec.api.split.FileSplit
import org.datavec.audio.recordreader.WavFileRecordReader
import java.io.File

class WavFileDL(
    val wavFileName: String = "some.wav"
) {

    fun dataSource(): RecordReader {
        val rr = WavFileRecordReader()
        val wavFile = File(wavFileName)
        val inputSplit = FileSplit(wavFile)
        rr.initialize(inputSplit)
        return rr
    }
}