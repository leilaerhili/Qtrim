package com.qtrim.android

object Profiles {
    val allowedIds: List<String> = listOf(
        "balanced",
        "high_fidelity",
        "low_latency",
        "low_cost",
    )

    private val labels: Map<String, String> = mapOf(
        "balanced" to "Balanced",
        "high_fidelity" to "High Fidelity",
        "low_latency" to "Low Latency",
        "low_cost" to "Low Cost",
    )

    private val aliases: Map<String, String> = mapOf(
        "default" to "balanced",
        "low_noise" to "high_fidelity",
        "noise" to "high_fidelity",
        "min_cx" to "high_fidelity",
        "few_cx" to "high_fidelity",
        "cx" to "high_fidelity",
        "latency" to "low_latency",
        "cost" to "low_cost",
    )

    fun labelFor(profileId: String): String {
        return labels[normalize(profileId)] ?: profileId
    }

    fun normalize(profileId: String): String {
        val key = profileId.trim().lowercase().replace(" ", "_")
        return aliases[key] ?: key
    }

    fun validate(profileId: String): String {
        val normalized = normalize(profileId)
        require(allowedIds.contains(normalized)) {
            "Unsupported profile_id '$profileId'. Choose one of: ${allowedIds.joinToString(", ")}."
        }
        return normalized
    }

    fun toConstraintId(profileId: String): Int {
        return when (normalize(profileId)) {
            "balanced" -> 0
            "low_latency" -> 1
            "high_fidelity" -> 2
            "low_cost" -> 4
            else -> 0
        }
    }
}
