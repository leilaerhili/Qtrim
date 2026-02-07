package com.qtrim.android

import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.Response.Status
import org.json.JSONArray
import org.json.JSONObject

class QtrimServer(
    port: Int,
    private val policy: OnnxPolicy,
    private val profileStore: ProfileStore,
) : NanoHTTPD(port) {

    override fun serve(session: IHTTPSession): Response {
        return try {
            when (session.method) {
                Method.GET -> handleGet(session)
                Method.POST -> handlePost(session)
                else -> jsonError(Status.METHOD_NOT_ALLOWED, "Unsupported method")
            }
        } catch (exc: Exception) {
            jsonError(Status.INTERNAL_ERROR, exc.message ?: "Unexpected error")
        }
    }

    private fun handleGet(session: IHTTPSession): Response {
        return when (session.uri) {
            "/health" -> jsonResponse(
                JSONObject(
                    mapOf(
                        "ok" to true,
                        "profile_id" to profileStore.getProfileId(),
                    )
                )
            )
            "/profile" -> jsonResponse(
                JSONObject(
                    mapOf(
                        "profile_id" to profileStore.getProfileId(),
                        "allowed_profiles" to Profiles.allowedIds,
                    )
                )
            )
            else -> jsonError(Status.NOT_FOUND, "Unknown endpoint")
        }
    }

    private fun handlePost(session: IHTTPSession): Response {
        return when (session.uri) {
            "/profile" -> handleProfileUpdate(session)
            "/infer" -> handleInfer(session)
            else -> jsonError(Status.NOT_FOUND, "Unknown endpoint")
        }
    }

    private fun handleProfileUpdate(session: IHTTPSession): Response {
        val payload = parseJsonBody(session) ?: return jsonError(Status.BAD_REQUEST, "Invalid JSON")
        val profileId = payload.optString("profile_id", "")
        return try {
            val updated = profileStore.setProfileId(profileId)
            jsonResponse(
                JSONObject(
                    mapOf(
                        "profile_id" to updated,
                        "allowed_profiles" to Profiles.allowedIds,
                    )
                )
            )
        } catch (exc: Exception) {
            jsonError(Status.BAD_REQUEST, exc.message ?: "Invalid profile_id")
        }
    }

    private fun handleInfer(session: IHTTPSession): Response {
        val payload = parseJsonBody(session) ?: return jsonError(Status.BAD_REQUEST, "Invalid JSON")
        val observationVector = parseObservationVector(payload)
            ?: return jsonError(Status.BAD_REQUEST, "Missing observation_vector")
        if (observationVector.size != 6) {
            return jsonError(Status.BAD_REQUEST, "observation_vector must have 6 floats")
        }
        val actionMask = parseActionMask(payload)
        val actionId = policy.predictAction(observationVector, actionMask)
        return jsonResponse(JSONObject(mapOf("action_id" to actionId)))
    }

    private fun parseObservationVector(payload: JSONObject): FloatArray? {
        val vec = payload.optJSONArray("observation_vector")
        if (vec != null) {
            return toFloatArray(vec)
        }
        val obs = payload.optJSONObject("observation") ?: return null
        return observationVectorFromPayload(obs)
    }

    private fun observationVectorFromPayload(obs: JSONObject): FloatArray {
        val gateCount = obs.optInt("gate_count", 0)
        val depth = obs.optInt("depth", 0)
        val numCnot = obs.optInt("num_cnot", obs.optInt("cx_count", 0))
        val numRz = obs.optInt("num_rz", 0)
        val lastActionId = obs.optInt("last_action_id", 0)
        val profile = obs.optString("priority_profile_id", obs.optString("constraint_profile", "balanced"))
        val profileId = Profiles.toConstraintId(profile)
        return floatArrayOf(
            gateCount.toFloat(),
            depth.toFloat(),
            numCnot.toFloat(),
            numRz.toFloat(),
            lastActionId.toFloat(),
            profileId.toFloat(),
        )
    }

    private fun parseActionMask(payload: JSONObject): IntArray? {
        val maskArray = payload.optJSONArray("action_mask") ?: return null
        return toIntArray(maskArray)
    }

    private fun toFloatArray(array: JSONArray): FloatArray {
        val out = FloatArray(array.length())
        for (i in 0 until array.length()) {
            out[i] = array.optDouble(i, 0.0).toFloat()
        }
        return out
    }

    private fun toIntArray(array: JSONArray): IntArray {
        val out = IntArray(array.length())
        for (i in 0 until array.length()) {
            out[i] = array.optInt(i, 0)
        }
        return out
    }

    private fun parseJsonBody(session: IHTTPSession): JSONObject? {
        return try {
            val body = HashMap<String, String>()
            session.parseBody(body)
            val raw = body["postData"] ?: ""
            if (raw.isBlank()) null else JSONObject(raw)
        } catch (exc: Exception) {
            null
        }
    }

    private fun jsonResponse(payload: JSONObject): Response {
        return newFixedLengthResponse(Status.OK, "application/json", payload.toString())
    }

    private fun jsonError(status: Status, message: String): Response {
        val payload = JSONObject(mapOf("error" to message))
        return newFixedLengthResponse(status, "application/json", payload.toString())
    }
}
