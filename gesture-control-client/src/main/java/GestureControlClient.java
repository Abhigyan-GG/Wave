import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.util.Base64;
import java.util.concurrent.*;
import org.json.*;
import com.github.sarxos.webcam.*;
import javax.imageio.ImageIO;

public class GestureControlClient extends JFrame {
    // Configuration
    private static final String SERVER_HOST = "localhost";
    private static final int SERVER_PORT = 8765;
    private static final int FRAME_WIDTH = 1024;
    private static final int FRAME_HEIGHT = 600;
    private static final int CAM_WIDTH = 640;
    private static final int CAM_HEIGHT = 480;
    private static final int UI_WIDTH = FRAME_WIDTH - CAM_WIDTH;
    private static final int PANEL_PADDING = 20;
    private static final int GESTURE_DISPLAY_TIME = 2000; // Show gesture label for 2 seconds

    // Colors
    private static final Color BG_COLOR = new Color(30, 30, 40);
    private static final Color PRIMARY_COLOR = new Color(76, 175, 80);
    private static final Color ACCENT_COLOR = new Color(255, 193, 7);
    private static final Color TEXT_COLOR = new Color(255, 255, 255);
    private static final Color INACTIVE_COLOR = new Color(100, 100, 100);

    // UI Components
    private JPanel cameraPanel;
    private JPanel controlPanel;
    private JProgressBar fanMeter;
    private JProgressBar volumeMeter;
    private JButton toggleFanButton;
    private JButton increaseButton;
    private JButton decreaseButton;
    private JLabel fanStatusLabel;
    private JLabel volumeStatusLabel;
    private JLabel connectionStatusLabel;
    private JLabel fpsLabel;
    private JLabel gestureLabel;

    // State
    private boolean fanState = false;
    private int fanSpeed = 100;
    private int lastFanSpeed = 100;
    private int volumeLevel = 50;
    private int currentFps = 0;
    private String lastGesture = "";
    private long lastGestureTime = 0;

    // WebSocket
    private WebSocket webSocket;
    private boolean connected = false;
    private ScheduledExecutorService scheduler;

    // Webcam
    private Webcam webcam;
    private WebcamPanel webcamPanel;
    private boolean isStreaming = false;
    private BufferedImage currentFrame;
    private final Object frameLock = new Object();

    public GestureControlClient() {
        super("Gesture Control System");
        setSize(FRAME_WIDTH, FRAME_HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        // Initialize components
        initializeUI();

        // Connect to WebSocket
        connectWebSocket();

        // Initialize webcam
        initializeWebcam();

        // Start frame sender
        startFrameSender();

        setVisible(true);
    }

    private void initializeUI() {
        // Camera panel (left side)
        cameraPanel = new JPanel();
        cameraPanel.setPreferredSize(new Dimension(CAM_WIDTH, FRAME_HEIGHT));
        cameraPanel.setBackground(Color.BLACK);
        cameraPanel.setLayout(new BorderLayout());

        // FPS and Gesture labels
        JPanel overlayPanel = new JPanel(new BorderLayout());
        overlayPanel.setOpaque(false);

        fpsLabel = new JLabel("FPS: 0");
        fpsLabel.setForeground(Color.YELLOW);
        fpsLabel.setFont(new Font("SansSerif", Font.BOLD, 14));
        fpsLabel.setBorder(BorderFactory.createEmptyBorder(10, 10, 0, 0));
        overlayPanel.add(fpsLabel, BorderLayout.NORTH);

        gestureLabel = new JLabel("");
        gestureLabel.setForeground(ACCENT_COLOR);
        gestureLabel.setFont(new Font("SansSerif", Font.BOLD, 18));
        gestureLabel.setHorizontalAlignment(JLabel.CENTER);
        overlayPanel.add(gestureLabel, BorderLayout.SOUTH);

        cameraPanel.add(overlayPanel, BorderLayout.CENTER);

        // Control panel (right side)
        controlPanel = new JPanel();
        controlPanel.setPreferredSize(new Dimension(UI_WIDTH, FRAME_HEIGHT));
        controlPanel.setBackground(BG_COLOR);
        controlPanel.setLayout(null); // Absolute positioning

        // Title
        JLabel titleLabel = new JLabel("Gesture Control");
        titleLabel.setForeground(TEXT_COLOR);
        titleLabel.setFont(new Font("SansSerif", Font.BOLD, 24));
        titleLabel.setBounds(PANEL_PADDING, 10, UI_WIDTH - 2 * PANEL_PADDING, 30);
        controlPanel.add(titleLabel);

        // Fan meter
        JLabel fanLabel = new JLabel("Fan Control (Right Hand)");
        fanLabel.setForeground(TEXT_COLOR);
        fanLabel.setFont(new Font("SansSerif", Font.PLAIN, 14));
        fanLabel.setBounds(PANEL_PADDING, 50, UI_WIDTH - 2 * PANEL_PADDING, 20);
        controlPanel.add(fanLabel);

        fanMeter = new JProgressBar(JProgressBar.VERTICAL, 0, 100);
        fanMeter.setValue(fanSpeed);
        fanMeter.setForeground(INACTIVE_COLOR);
        fanMeter.setBackground(new Color(50, 50, 60));
        fanMeter.setBounds(PANEL_PADDING, 70, UI_WIDTH - 2 * PANEL_PADDING, 150);
        fanMeter.setBorder(BorderFactory.createEmptyBorder());
        controlPanel.add(fanMeter);

        fanStatusLabel = new JLabel("Fan: STOPPED");
        fanStatusLabel.setForeground(INACTIVE_COLOR);
        fanStatusLabel.setFont(new Font("SansSerif", Font.BOLD, 16));
        fanStatusLabel.setBounds(PANEL_PADDING, 230, UI_WIDTH - 2 * PANEL_PADDING, 20);
        controlPanel.add(fanStatusLabel);

        // Volume meter
        JLabel volumeLabel = new JLabel("Volume Control (Left Hand)");
        volumeLabel.setForeground(TEXT_COLOR);
        volumeLabel.setFont(new Font("SansSerif", Font.PLAIN, 14));
        volumeLabel.setBounds(PANEL_PADDING, 250, UI_WIDTH - 2 * PANEL_PADDING, 20);
        controlPanel.add(volumeLabel);

        volumeMeter = new JProgressBar(JProgressBar.VERTICAL, 0, 100);
        volumeMeter.setValue(volumeLevel);
        volumeMeter.setForeground(PRIMARY_COLOR);
        volumeMeter.setBackground(new Color(50, 50, 60));
        volumeMeter.setBounds(PANEL_PADDING, 270, UI_WIDTH - 2 * PANEL_PADDING, 150);
        volumeMeter.setBorder(BorderFactory.createEmptyBorder());
        controlPanel.add(volumeMeter);

        volumeStatusLabel = new JLabel("System Volume: 50%");
        volumeStatusLabel.setForeground(PRIMARY_COLOR);
        volumeStatusLabel.setFont(new Font("SansSerif", Font.BOLD, 16));
        volumeStatusLabel.setBounds(PANEL_PADDING, 430, UI_WIDTH - 2 * PANEL_PADDING, 20);
        controlPanel.add(volumeStatusLabel);

        // Control buttons
        increaseButton = new JButton("+");
        increaseButton.setBounds(PANEL_PADDING, 470, 60, 40);
        increaseButton.setEnabled(false);
        increaseButton.addActionListener(e -> adjustFanSpeed(25));
        controlPanel.add(increaseButton);

        decreaseButton = new JButton("-");
        decreaseButton.setBounds(PANEL_PADDING + 70, 470, 60, 40);
        decreaseButton.setEnabled(false);
        decreaseButton.addActionListener(e -> adjustFanSpeed(-25));
        controlPanel.add(decreaseButton);

        toggleFanButton = new JButton("TOGGLE FAN");
        toggleFanButton.setBounds(PANEL_PADDING, 520, UI_WIDTH - 2 * PANEL_PADDING, 50);
        toggleFanButton.setBackground(INACTIVE_COLOR);
        toggleFanButton.setForeground(TEXT_COLOR);
        toggleFanButton.addActionListener(e -> toggleFan());
        controlPanel.add(toggleFanButton);

        // Connection status
        connectionStatusLabel = new JLabel("Connection: Disconnected");
        connectionStatusLabel.setForeground(ACCENT_COLOR);
        connectionStatusLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));
        connectionStatusLabel.setBounds(PANEL_PADDING, FRAME_HEIGHT - 50, UI_WIDTH - 2 * PANEL_PADDING, 20);
        controlPanel.add(connectionStatusLabel);

        // Add panels to frame
        add(cameraPanel, BorderLayout.WEST);
        add(controlPanel, BorderLayout.EAST);
    }

    private void initializeWebcam() {
        try {
            webcam = Webcam.getDefault();
            if (webcam != null) {
                webcam.setViewSize(new Dimension(CAM_WIDTH, CAM_HEIGHT));
                webcam.open();

                webcamPanel = new WebcamPanel(webcam);
                webcamPanel.setFPSDisplayed(false);
                webcamPanel.setDisplayDebugInfo(false);
                webcamPanel.setImageSizeDisplayed(false);
                webcamPanel.setMirrored(true);

                cameraPanel.add(webcamPanel, BorderLayout.CENTER);

                // Start frame capturing thread
                new Thread(() -> {
                    while (true) {
                        if (webcam.isOpen() && connected) {
                            BufferedImage frame = webcam.getImage();
                            if (frame != null) {
                                synchronized (frameLock) {
                                    currentFrame = frame;
                                }
                            }
                        }
                        try {
                            Thread.sleep(33); // ~30 FPS
                        } catch (InterruptedException e) {
                            break;
                        }
                    }
                }).start();

                isStreaming = true;
            } else {
                JLabel errorLabel = new JLabel("No webcam detected!");
                errorLabel.setForeground(Color.RED);
                errorLabel.setHorizontalAlignment(JLabel.CENTER);
                cameraPanel.add(errorLabel, BorderLayout.CENTER);
            }
        } catch (Exception e) {
            e.printStackTrace();
            JLabel errorLabel = new JLabel("Error initializing webcam: " + e.getMessage());
            errorLabel.setForeground(Color.RED);
            errorLabel.setHorizontalAlignment(JLabel.CENTER);
            cameraPanel.add(errorLabel, BorderLayout.CENTER);
        }
    }

    private void connectWebSocket() {
        try {
            HttpClient client = HttpClient.newHttpClient();
            WebSocket.Listener listener = new WebSocketListener();

            CompletableFuture<WebSocket> futureWs = client.newWebSocketBuilder()
                    .buildAsync(URI.create("ws://" + SERVER_HOST + ":" + SERVER_PORT), listener);

            webSocket = futureWs.get(5, TimeUnit.SECONDS);

            // Start a scheduler to check connection status
            scheduler = Executors.newSingleThreadScheduledExecutor();
            scheduler.scheduleAtFixedRate(() -> {
                if (webSocket != null) {
                    try {
                        // Send a heartbeat to keep connection alive
                        webSocket.sendText("{\"type\":\"heartbeat\"}", true);
                    } catch (Exception e) {
                        connected = false;
                        updateConnectionStatus();
                        // Try to reconnect
                        reconnectWebSocket();
                    }
                }
            }, 5, 5, TimeUnit.SECONDS);

        } catch (Exception e) {
            e.printStackTrace();
            connectionStatusLabel.setText("Connection: Failed - " + e.getMessage());
            // Schedule reconnection attempt
            if (scheduler == null) {
                scheduler = Executors.newSingleThreadScheduledExecutor();
            }
            scheduler.schedule(this::reconnectWebSocket, 5, TimeUnit.SECONDS);
        }
    }

    private void reconnectWebSocket() {
        SwingUtilities.invokeLater(() -> {
            connectionStatusLabel.setText("Connection: Reconnecting...");
        });

        try {
            if (webSocket != null) {
                try {
                    webSocket.abort();
                } catch (Exception ignored) {}
            }

            HttpClient client = HttpClient.newHttpClient();
            WebSocket.Listener listener = new WebSocketListener();

            CompletableFuture<WebSocket> futureWs = client.newWebSocketBuilder()
                    .buildAsync(URI.create("ws://" + SERVER_HOST + ":" + SERVER_PORT), listener);

            webSocket = futureWs.get(5, TimeUnit.SECONDS);
        } catch (Exception e) {
            e.printStackTrace();
            SwingUtilities.invokeLater(() -> {
                connectionStatusLabel.setText("Connection: Failed - Retrying in 5s");
            });

            scheduler.schedule(this::reconnectWebSocket, 5, TimeUnit.SECONDS);
        }
    }

    private void startFrameSender() {
        ScheduledExecutorService frameScheduler = Executors.newSingleThreadScheduledExecutor();
        frameScheduler.scheduleAtFixedRate(() -> {
            if (connected && isStreaming && webcam != null && webcam.isOpen()) {
                try {
                    BufferedImage frame;
                    synchronized (frameLock) {
                        if (currentFrame == null) return;
                        frame = currentFrame;
                    }

                    // Convert frame to JPEG
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    ImageIO.write(frame, "jpg", baos);
                    byte[] jpegData = baos.toByteArray();

                    // Send frame to server
                    if (webSocket != null) {
                        byte[] header = "frame:".getBytes();
                        ByteBuffer buffer = ByteBuffer.allocate(header.length + jpegData.length);
                        buffer.put(header);
                        buffer.put(jpegData);
                        buffer.flip();

                        webSocket.sendBinary(buffer, true).join();
                    }
                } catch (Exception e) {
                    System.err.println("Error sending frame: " + e.getMessage());
                }
            }
        }, 0, 100, TimeUnit.MILLISECONDS); // Send frames at 10fps
    }

    private void updateUI(JSONObject state) {
        SwingUtilities.invokeLater(() -> {
            try {
                if (state.has("fan_state")) {
                    fanState = state.getBoolean("fan_state");
                }

                if (state.has("fan_speed")) {
                    fanSpeed = state.getInt("fan_speed");
                    if (!fanState) {
                        lastFanSpeed = fanSpeed > 0 ? fanSpeed : lastFanSpeed;
                    }
                }

                if (state.has("volume")) {
                    volumeLevel = state.getInt("volume");
                }

                // Update UI components
                fanMeter.setValue(fanSpeed);
                fanMeter.setForeground(fanState ? PRIMARY_COLOR : INACTIVE_COLOR);
                fanStatusLabel.setText("Fan: " + (fanState ? "RUNNING" : "STOPPED"));
                fanStatusLabel.setForeground(fanState ? PRIMARY_COLOR : INACTIVE_COLOR);

                volumeMeter.setValue(volumeLevel);
                volumeStatusLabel.setText("System Volume: " + volumeLevel + "%");

                toggleFanButton.setBackground(fanState ? PRIMARY_COLOR : INACTIVE_COLOR);

                increaseButton.setEnabled(fanState);
                decreaseButton.setEnabled(fanState);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    private void updateConnectionStatus() {
        SwingUtilities.invokeLater(() -> {
            connectionStatusLabel.setText("Connection: " + (connected ? "Connected" : "Disconnected"));
            connectionStatusLabel.setForeground(connected ? PRIMARY_COLOR : Color.RED);
        });
    }

    private void updateGestureDisplay(String gesture) {
        SwingUtilities.invokeLater(() -> {
            lastGesture = gesture;
            lastGestureTime = System.currentTimeMillis();

            // Make gesture name more readable
            String displayText = "";
            switch (gesture) {
                case "palm-open":
                    displayText = "PALM OPEN - Fan ON";
                    break;
                case "palm-closed":
                    displayText = "PALM CLOSED - Fan OFF";
                    break;
                case "pinch":
                    displayText = "PINCH - Decrease";
                    break;
                case "spread":
                    displayText = "SPREAD - Increase";
                    break;
                default:
                    displayText = gesture.toUpperCase();
            }

            gestureLabel.setText(displayText);

            // Start a timer to clear the gesture after 2 seconds
            Timer timer = new Timer(GESTURE_DISPLAY_TIME, e -> {
                if (lastGesture.equals(gesture) &&
                        System.currentTimeMillis() - lastGestureTime >= GESTURE_DISPLAY_TIME) {
                    gestureLabel.setText("");
                }
            });
            timer.setRepeats(false);
            timer.start();
        });
    }

    private void toggleFan() {
        try {
            fanState = !fanState;

            // Update fan speed based on state
            if (fanState) {
                fanSpeed = lastFanSpeed > 0 ? lastFanSpeed : 100;
            } else {
                lastFanSpeed = fanSpeed;
                fanSpeed = 0;
            }

            // Send command to server
            JSONObject command = new JSONObject();
            command.put("command", "set_fan");
            command.put("state", fanState);
            command.put("value", fanSpeed);

            webSocket.sendText(command.toString(), true);

            // Update UI immediately for responsiveness
            updateUI(command);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void adjustFanSpeed(int delta) {
        try {
            int newSpeed = Math.max(0, Math.min(100, fanSpeed + delta));
            if (newSpeed != fanSpeed) {
                fanSpeed = newSpeed;

                // Send command to server
                JSONObject command = new JSONObject();
                command.put("command", "set_fan");
                command.put("state", fanState);
                command.put("value", fanSpeed);

                webSocket.sendText(command.toString(), true);

                // Update UI immediately
                updateUI(command);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private class WebSocketListener implements WebSocket.Listener {
        private StringBuilder textBuffer = new StringBuilder();

        @Override
        public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
            textBuffer.append(data);

            if (last) {
                try {
                    String message = textBuffer.toString();
                    JSONObject json = new JSONObject(message);

                    String type = json.optString("type", "");

                    if ("state".equals(type)) {
                        // State update
                        updateUI(json);
                    } else if ("gesture".equals(type)) {
                        // Gesture detected
                        String gestureName = json.optString("name", "");
                        if (!gestureName.isEmpty()) {
                            updateGestureDisplay(gestureName);
                        }
                    }
                } catch (JSONException e) {
                    // Not JSON or invalid format, ignore
                    System.err.println("Received non-JSON text: " + textBuffer);
                }

                textBuffer = new StringBuilder();
            }

            return WebSocket.Listener.super.onText(webSocket, data, last);
        }

        @Override
        public void onOpen(WebSocket webSocket) {
            connected = true;
            updateConnectionStatus();
            WebSocket.Listener.super.onOpen(webSocket);
        }

        @Override
        public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
            connected = false;
            updateConnectionStatus();
            // Try to reconnect
            scheduler.schedule(GestureControlClient.this::reconnectWebSocket, 5, TimeUnit.SECONDS);
            return WebSocket.Listener.super.onClose(webSocket, statusCode, reason);
        }

        @Override
        public void onError(WebSocket webSocket, Throwable error) {
            error.printStackTrace();
            connected = false;
            updateConnectionStatus();
            WebSocket.Listener.super.onError(webSocket, error);
        }
    }

    public static void main(String[] args) {
        try {
            // Set system look and feel
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }

        SwingUtilities.invokeLater(() -> {
            new GestureControlClient();
        });
    }
}