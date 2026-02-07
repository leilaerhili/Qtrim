package com.qtrim.android

class ProfileStore(initialProfileId: String) {
    @Volatile
    private var currentProfileId: String = try {
        Profiles.validate(initialProfileId)
    } catch (_: Exception) {
        "balanced"
    }

    fun getProfileId(): String {
        return currentProfileId
    }

    @Synchronized
    fun setProfileId(profileId: String): String {
        val normalized = Profiles.validate(profileId)
        currentProfileId = normalized
        return normalized
    }
}
