import asyncio
import websockets
import cv2
import numpy as np
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ws_test_client")


async def test_client():
    uri = "ws://localhost:8765"
    logger.info(f"Connecting to {uri} ...")
    try:
        async with websockets.connect(uri, ping_interval=20) as websocket:
            logger.info("Connected to the server.")
            # Send an initial greeting if desired.
            await websocket.send("Hello, server!")
            logger.info("Sent initial greeting to server.")

            while True:
                try:
                    message = await websocket.recv()
                except websockets.ConnectionClosed as e:
                    logger.error(f"Connection closed: {e}")
                    break

                # If the message is in bytes, check for the camera frame header.
                if isinstance(message, bytes):
                    header = b"frame:"
                    if message.startswith(header):
                        jpeg_bytes = message[len(header):]
                        nparr = np.frombuffer(jpeg_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            cv2.imshow("Camera Feed", frame)
                            # Press "q" to exit.
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        else:
                            logger.error("Failed to decode frame.")
                    else:
                        logger.warning("Received unknown binary message.")
                else:
                    # Try to decode JSON state updates.
                    try:
                        data = json.loads(message)
                        logger.info(f"Received state update: {data}")
                    except Exception as ex:
                        logger.info(f"Received text message: {message}")
    except Exception as ex:
        logger.exception(f"An error occurred: {ex}")
    finally:
        cv2.destroyAllWindows()
        logger.info("Client finished.")


if __name__ == "__main__":
    asyncio.run(test_client())
