package com.qtrim.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer

class OnnxPolicy private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String,
    private val outputName: String,
) {
    fun predictAction(observationVector: FloatArray, actionMask: IntArray?): Int {
        val shape = longArrayOf(1, observationVector.size.toLong())
        val inputBuffer = FloatBuffer.wrap(observationVector)

        OnnxTensor.createTensor(env, inputBuffer, shape).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor)).use { result ->
                val raw = result[0].value
                val qValues = when (raw) {
                    is Array<*> -> (raw[0] as? FloatArray) ?: return 0
                    is FloatArray -> raw
                    else -> return 0
                }
                val masked = if (actionMask != null && actionMask.size == qValues.size) {
                    val tmp = qValues.clone()
                    for (i in actionMask.indices) {
                        if (actionMask[i] <= 0) {
                            tmp[i] = Float.NEGATIVE_INFINITY
                        }
                    }
                    tmp
                } else {
                    qValues
                }
                var bestIdx = 0
                var bestVal = masked[0]
                for (i in 1 until masked.size) {
                    if (masked[i] > bestVal) {
                        bestVal = masked[i]
                        bestIdx = i
                    }
                }
                return bestIdx
            }
        }
    }

    companion object {
        fun loadFromAssets(context: Context, assetName: String): OnnxPolicy {
            val modelFile = ensureModelFile(context, assetName)
            val env = OrtEnvironment.getEnvironment()
            val session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
            val inputName = session.inputNames.first()
            val outputName = session.outputNames.first()
            return OnnxPolicy(env, session, inputName, outputName)
        }

        private fun ensureModelFile(context: Context, assetName: String): File {
            val outFile = File(context.filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) {
                return outFile
            }
            context.assets.open(assetName).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            return outFile
        }
    }
}
