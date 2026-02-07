package com.qtrim.android

import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import fi.iki.elonen.NanoHTTPD
import java.net.Inet4Address
import java.net.NetworkInterface

class MainActivity : AppCompatActivity() {
    private lateinit var statusValue: TextView
    private lateinit var endpointValue: TextView
    private lateinit var profileValue: TextView
    private lateinit var profileSpinner: Spinner
    private lateinit var restartButton: Button

    private val profileStore = ProfileStore("balanced")
    private var policy: OnnxPolicy? = null
    private var server: QtrimServer? = null

    private val serverPort = 9002
    private val assetModelName = "tiny_infer_handoff_seed0_android_int8_bs1.onnx"
    private val profileOptions = Profiles.allowedIds
    private val profileLabels = profileOptions.map { Profiles.labelFor(it) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusValue = findViewById(R.id.statusValue)
        endpointValue = findViewById(R.id.endpointValue)
        profileValue = findViewById(R.id.profileValue)
        profileSpinner = findViewById(R.id.profileSpinner)
        restartButton = findViewById(R.id.restartButton)

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, profileLabels)
        profileSpinner.adapter = adapter
        profileSpinner.setSelection(profileOptions.indexOf(profileStore.getProfileId()))
        profileSpinner.setOnItemSelectedListener(SimpleItemSelectedListener { position ->
            val selectedId = profileOptions.getOrNull(position) ?: "balanced"
            val updated = profileStore.setProfileId(selectedId)
            updateProfileValue(updated)
        })

        restartButton.setOnClickListener {
            restartServer()
        }

        updateProfileValue(profileStore.getProfileId())
        updateEndpointText()
        loadPolicyAsync()
    }

    override fun onStart() {
        super.onStart()
        startServerIfReady()
    }

    override fun onStop() {
        super.onStop()
        stopServer()
    }

    private fun loadPolicyAsync() {
        updateStatus("Loading model...")
        Thread {
            try {
                val loaded = OnnxPolicy.loadFromAssets(this, assetModelName)
                policy = loaded
                runOnUiThread {
                    updateStatus("Model loaded.")
                    startServerIfReady()
                }
            } catch (exc: Exception) {
                runOnUiThread {
                    updateStatus("Model load failed: ${exc.message}")
                    Toast.makeText(this, "Model load failed", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun startServerIfReady() {
        if (server != null) {
            return
        }
        val currentPolicy = policy ?: run {
            updateStatus("Waiting for model...")
            return
        }
        val newServer = QtrimServer(serverPort, currentPolicy, profileStore)
        try {
            newServer.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false)
            server = newServer
            updateStatus("Running on port $serverPort")
            updateEndpointText()
        } catch (exc: Exception) {
            updateStatus("Server failed: ${exc.message}")
        }
    }

    private fun stopServer() {
        server?.stop()
        server = null
        updateStatus("Stopped")
    }

    private fun restartServer() {
        stopServer()
        startServerIfReady()
    }

    private fun updateStatus(message: String) {
        statusValue.text = message
    }

    private fun updateProfileValue(profileId: String) {
        val label = Profiles.labelFor(profileId)
        profileValue.text = "Current: $label ($profileId)"
    }

    private fun updateEndpointText() {
        val ip = getDeviceIpAddress()
        endpointValue.text = "http://$ip:$serverPort/infer"
    }

    private fun getDeviceIpAddress(): String {
        return try {
            val interfaces = NetworkInterface.getNetworkInterfaces()
            for (iface in interfaces) {
                val addresses = iface.inetAddresses
                for (addr in addresses) {
                    if (!addr.isLoopbackAddress && addr is Inet4Address) {
                        return addr.hostAddress ?: "127.0.0.1"
                    }
                }
            }
            "127.0.0.1"
        } catch (_: Exception) {
            "127.0.0.1"
        }
    }
}
