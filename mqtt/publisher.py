"""
mqtt/publisher.py
=================
MQTT publisher that sends OCR results to the EMQX broker.

Topics used
-----------
industrial/ocr/results  — JSON OCR result payload
industrial/ocr/status   — heartbeat / connection status messages
industrial/ocr/image    — reserved for future camera frame metadata

QoS = 1 (at-least-once delivery) — balances reliability vs overhead.

Design
------
- MQTTPublisher wraps paho-mqtt with automatic reconnect logic.
- publish_result() accepts an OCRResult dataclass or a plain dict.
- Thread-safe: paho's network loop runs in its own thread.
- Graceful disconnect on context exit (use as a context manager).

Usage
-----
    from mqtt.publisher import MQTTPublisher
    from inference.pipeline import OCRResult

    with MQTTPublisher() as pub:
        pub.publish_result(ocr_result)
        pub.publish_status("online")
"""

import json
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Union

import paho.mqtt.client as mqtt
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


class MQTTPublisher:
    """
    Paho-MQTT publisher configured for EMQX.

    Parameters
    ----------
    host, port, username, password, client_id
        Connection credentials — all defaulted from config.py.
    """

    def __init__(
        self,
        host: str = settings.MQTT_BROKER_HOST,
        port: int = settings.MQTT_BROKER_PORT,
        username: str = settings.MQTT_USERNAME,
        password: str = settings.MQTT_PASSWORD,
        client_id: str = settings.MQTT_CLIENT_ID_PUB,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._connected = threading.Event()

        self._client = mqtt.Client(
            client_id=client_id,
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )

        if username:
            self._client.username_pw_set(username, password)

        # Callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish

        # Will message — broker broadcasts this if client disconnects unexpectedly
        self._client.will_set(
            settings.MQTT_TOPIC_STATUS,
            payload=json.dumps({"client": client_id, "status": "offline"}),
            qos=settings.MQTT_QOS,
            retain=True,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            logger.success(f"MQTT publisher connected to {self._host}:{self._port}")
            self._connected.set()
            # Publish online status
            client.publish(
                settings.MQTT_TOPIC_STATUS,
                json.dumps({"client": self._client_id, "status": "online"}),
                qos=settings.MQTT_QOS,
                retain=True,
            )
        else:
            logger.error(f"MQTT connect failed. Reason code: {reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        self._connected.clear()
        logger.warning(f"MQTT publisher disconnected (rc={reason_code}). Reconnecting...")

    def _on_publish(self, client, userdata, mid, reason_code=None, properties=None):
        logger.debug(f"MQTT message published (mid={mid})")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect(self, timeout: int = 10) -> None:
        """Connect to broker and wait until the connection is established."""
        logger.info(f"Connecting MQTT publisher to {self._host}:{self._port}...")
        try:
            self._client.connect(
                self._host, self._port, keepalive=settings.MQTT_KEEPALIVE
            )
            self._client.loop_start()   # background thread

            if not self._connected.wait(timeout=timeout):
                raise TimeoutError(
                    f"MQTT connection timed out after {timeout}s. "
                    f"Is EMQX running at {self._host}:{self._port}?"
                )
        except Exception as exc:
            logger.error(f"MQTT publisher connection error: {exc}")
            raise

    def disconnect(self) -> None:
        """Gracefully disconnect from the broker."""
        self._client.publish(
            settings.MQTT_TOPIC_STATUS,
            json.dumps({"client": self._client_id, "status": "offline"}),
            qos=settings.MQTT_QOS,
            retain=True,
        )
        time.sleep(0.2)   # let the will message flush
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("MQTT publisher disconnected.")

    # ------------------------------------------------------------------
    # Publishing helpers
    # ------------------------------------------------------------------
    def publish(self, topic: str, payload: dict, qos: int = settings.MQTT_QOS) -> None:
        """
        Publish a dict as JSON to *topic*.

        Raises RuntimeError if not connected.
        """
        if not self._connected.is_set():
            raise RuntimeError("MQTT publisher is not connected. Call connect() first.")

        message = json.dumps(payload, default=str)
        result = self._client.publish(topic, message, qos=qos)

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(f"Publish to '{topic}' failed (rc={result.rc})")
        else:
            logger.info(f"Published to '{topic}': {message[:120]}...")

    def publish_result(self, result) -> None:
        """
        Publish an OCRResult (dataclass or dict) to the results topic.

        Parameters
        ----------
        result : OCRResult | dict
            The OCR result to publish.
        """
        if hasattr(result, "__dataclass_fields__"):
            payload = asdict(result)
        elif isinstance(result, dict):
            payload = result
        else:
            raise TypeError(f"Unsupported result type: {type(result)}")

        # Only send the fields needed by downstream consumers
        slim_payload = {
            "filename":   payload.get("filename", ""),
            "text":       payload.get("text", ""),
            "length":     payload.get("length", 0),
            "confidence": payload.get("confidence", 0.0),
            "status":     payload.get("status", ""),
        }
        self.publish(settings.MQTT_TOPIC_RESULTS, slim_payload)

    def publish_status(self, status: str) -> None:
        """Publish a freeform status string."""
        self.publish(
            settings.MQTT_TOPIC_STATUS,
            {"client": self._client_id, "status": status},
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False   # don't suppress exceptions
