package sample

import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.slf4j.LoggerFactory
import java.io.File


class SomeNLP {

    private val log = LoggerFactory.getLogger(SomeNLP::class.java)!!

    fun main() {

        // Gets Path to Text file
//        val filePath = ClassPathResource("raw_sentences.txt").file.absolutePath
        val filePath = File("raw_sentences.txt").absolutePath

        log.info("Load & Vectorize Sentences....")
        // Strip white space before and after for each line
        val iter = BasicLineIterator(filePath)
        // Split on white spaces in the line to get words
        val t = DefaultTokenizerFactory()

        /*
            CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.
         */
        t.tokenPreProcessor = CommonPreprocessor()

        log.info("Building model....")
        val vec = Word2Vec.Builder()
            .minWordFrequency(5)
            .iterations(3)
            .layerSize(1000)
            .seed(84)
            .windowSize(10)
            .iterate(iter)
            .tokenizerFactory(t)
            .build()

        log.info("Fitting Word2Vec model....")
        vec.fit()

        log.info("Writing word vectors to text file....")

        // Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
        log.info("Closest Words:")
        val lst = vec.wordsNearestSum("я", 10)
        log.info("10 Words closest to 'day': {}", lst)

        // TODO resolve missing UiServer
        //        UiServer server = UiServer.getInstance();
        //        System.out.println("Started on port " + server.getPort());
    }
}

