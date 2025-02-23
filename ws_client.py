import asyncio
import cv2
import numpy as np
import websockets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_client")

# WebSocket server URI (adjust if running on another machine)
WS_SERVER_URI = "ws://localhost:8765"

# Header to prepend to each frame
HEADER = b"frame:"


async def send_frames():
    while True:  # Keep retrying if connection is lost
        try:
            async with websockets.connect(WS_SERVER_URI) as websocket:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow to avoid MSMF issues on Windows
                if not cap.isOpened():
                    logger.error("Error: Could not open camera.")
                    return

                logger.info("Connected to WebSocket server. Starting frame transmission...")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to capture frame")
                        continue

                    # Encode the frame as JPEG
                    success, jpeg = cv2.imencode('.jpg', frame)
                    if not success:
                        logger.error("Failed to encode frame")
                        continue

                    # Construct the payload with a header
                    payload = HEADER + jpeg.tobytes()

                    # Send the frame to the WebSocket server
                    await websocket.send(payload)
                    logger.debug("Frame sent")

                    await asyncio.sleep(0.1)  # Control frame rate (~10 fps)

        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"WebSocket connection closed unexpectedly: {e}")
            logger.info("Reconnecting in 2 seconds...")
            await asyncio.sleep(2)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            await asyncio.sleep(2)


def main():
    asyncio.run(send_frames())


if __name__ == "__main__":
    main()
