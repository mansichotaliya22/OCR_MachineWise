"""
mqtt/subscriber.py
==================
MQTT subscriber that listens for OCR results from EMQX and persists
them to MongoDB via the database layer.

Topics subscribed
-----------------
industrial/ocr/results  — incoming OCR JSON payloads
industrial/ocr/status   — broker status messages (logged only)

Design
------
- Runs in a background thread (loop_forever).
- On each message on the results topic, parses the JSON and calls the
  provided callback (typically the MongoDB insert function).
- Optional on_message_callback lets callers inject custom handling
  (e.g. WebSocket broadcast in Phase 3).
- Automatic reconnect is handled by paho's built-in mechanism.
"""

import json
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import paho.mqtt.client as mqtt
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


class MQTTSubscriber:
    """
    Persistent MQTT subscriber for OCR results.

    Parameters
    ----------
    on_result_callback : Callable[[dict], None] | None
        Called with the parsed result dict whenever a message arrives
        on the results topic.  Pass your MongoDB insert function here.
    """

    def __init__(
        self,
        on_result_callback: Optional[Callable[[dict], None]] = None,
        host: str = settings.MQTT_BROKER_HOST,
        port: int = settings.MQTT_BROKER_PORT,
        username: str = settings.MQTT_USERNAME,
        password: str = settings.MQTT_PASSWORD,
        client_id: str = settings.MQTT_CLIENT_ID_SUB,
    ) -> None:
        self._host = host
        self._port = port
        self._on_result_callback = on_result_callback
        self._thread: Optional[threading.Thread] = None

        self._client = mqtt.Client(
            client_id=client_id,
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )

        if username:
            self._client.username_pw_set(username, password)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # Enable automatic reconnect
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            logger.success(
                f"MQTT subscriber connected to {self._host}:{self._port}"
            )
            # Subscribe to all OCR topics
            client.subscribe(settings.MQTT_TOPIC_RESULTS, qos=settings.MQTT_QOS)
            client.subscribe(settings.MQTT_TOPIC_STATUS,  qos=settings.MQTT_QOS)
            logger.info(
                f"Subscribed to: {settings.MQTT_TOPIC_RESULTS}, "
                f"{settings.MQTT_TOPIC_STATUS}"
            )
        else:
            logger.error(f"MQTT subscriber connect failed (rc={reason_code})")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        logger.warning(f"MQTT subscriber disconnected (rc={reason_code}). Auto-reconnecting...")

    def _on_message(self, client, userdata, message: mqtt.MQTTMessage):
        topic = message.topic
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Could not parse MQTT message on '{topic}': {e}")
            return

        logger.debug(f"MQTT message on '{topic}': {payload}")

        if topic == settings.MQTT_TOPIC_RESULTS:
            logger.info(
                f"OCR result received — text='{payload.get('text', '')}' "
                f"conf={payload.get('confidence', 0)}"
            )
            if self._on_result_callback:
                try:
                    self._on_result_callback(payload)
                except Exception as e:
                    logger.error(f"on_result_callback raised: {e}")

        elif topic == settings.MQTT_TOPIC_STATUS:
            logger.info(f"MQTT status update: {payload}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, timeout: int = 10) -> None:
        """
        Connect to broker and start listening in a background thread.
        Returns immediately — the listener runs in the background.
        """
        logger.info(f"Starting MQTT subscriber → {self._host}:{self._port}")
        self._client.connect(
            self._host, self._port, keepalive=settings.MQTT_KEEPALIVE
        )
        self._thread = threading.Thread(
            target=self._client.loop_forever,
            name="mqtt-subscriber",
            daemon=True,   # dies automatically when main process exits
        )
        self._thread.start()
        logger.info("MQTT subscriber running in background thread.")

    def stop(self) -> None:
        """Stop the subscriber cleanly."""
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("MQTT subscriber stopped.")

    def set_callback(self, callback: Callable[[dict], None]) -> None:
        """Replace the result callback at runtime."""
        self._on_result_callback = callback
